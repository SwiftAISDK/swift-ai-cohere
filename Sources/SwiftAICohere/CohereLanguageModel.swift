import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class CohereLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID = "cohere.chat"
    public let modelID: String
    private let config: CohereHTTPConfiguration

    public init(modelID: String, http: AIProviderHTTPClient) {
        self.modelID = modelID
        self.config = CohereHTTPConfiguration(http: http)
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try body(for: request, stream: false)
        let response = try await config.sendJSONResponse(
            path: "/chat",
            modelID: modelID,
            body: .object(prepared.body),
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let raw = response.json
        let text = raw["message"]?["content"]?.arrayValue?.compactMap { item in
            item["type"]?.stringValue == "text" ? item["text"]?.stringValue : nil
        }.joined() ?? ""
        let reasoning = raw["message"]?["content"]?.arrayValue?.compactMap { item in
            item["type"]?.stringValue == "thinking" ? item["thinking"]?.stringValue : nil
        }.joined() ?? ""
        let toolCalls = cohereToolCalls(from: raw["message"]?["tool_calls"])
        guard !text.isEmpty || !reasoning.isEmpty || raw["message"] != nil else {
            throw AIError.invalidResponse(provider: providerID, message: "No Cohere message content found.")
        }
        return TextGenerationResult(
            text: text,
            reasoning: reasoning,
            finishReason: mapCohereFinishReason(raw["finish_reason"]?.stringValue),
            usage: cohereTokenUsage(from: raw),
            toolCalls: toolCalls,
            sources: cohereSources(from: raw["message"]?["citations"]),
            rawValue: raw,
            warnings: prepared.warnings,
            responseMetadata: cohereResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try body(for: request, stream: true)
                    let httpRequest = try config.request(
                        path: "/chat",
                        modelID: modelID,
                        body: .object(prepared.body),
                        headers: request.headers,
                        abortSignal: request.abortSignal
                    )
                    let response = try await config.streamRequest(httpRequest)
                    guard (200..<300).contains(response.statusCode) else {
                        throw apiCallError(provider: providerID, response: try await bufferedHTTPResponse(from: response, request: httpRequest))
                    }
                    let responseHead = httpResponseHead(from: response, request: httpRequest)
                    continuation.yield(.streamStart(warnings: prepared.warnings))
                    continuation.yield(.responseMetadata(cohereResponseMetadata(response: responseHead, modelID: modelID)))
                    var finishReason: String? = "interrupted"
                    var usage: TokenUsage?
                    var pendingToolCall: CoherePendingToolCall?
                    var activeTextIDs: Set<String> = []
                    var activeReasoningIDs: Set<String> = []
                    eventLoop: for try await event in serverSentEvents(from: response.body) {
                        if event.data == "[DONE]" { break }
                        let raw: JSONValue
                        do {
                            raw = try decodeJSONBody(Data(event.data.utf8))
                        } catch {
                            finishReason = "error"
                            continuation.yield(.error(message: String(describing: error)))
                            continue
                        }
                        if request.includeRawChunks {
                            continuation.yield(.raw(raw))
                        }
                        switch raw["type"]?.stringValue {
                        case "message-start":
                            let metadataRaw: JSONValue? = raw["id"].map { .object(["id": $0]) }
                            continuation.yield(.responseMetadata(cohereResponseMetadata(from: metadataRaw, response: responseHead, modelID: modelID)))
                        case "content-start":
                            let id = String(raw["index"]?.intValue ?? 0)
                            if raw["delta"]?["message"]?["content"]?["type"]?.stringValue == "thinking" {
                                if activeReasoningIDs.insert(id).inserted {
                                    continuation.yield(.reasoningStart(id: id))
                                }
                            } else {
                                if activeTextIDs.insert(id).inserted {
                                    continuation.yield(.textStart(id: id))
                                }
                            }
                        case "content-delta":
                            let id = String(raw["index"]?.intValue ?? 0)
                            if let text = raw["delta"]?["message"]?["content"]?["text"]?.stringValue {
                                if activeTextIDs.insert(id).inserted {
                                    continuation.yield(.textStart(id: id))
                                }
                                continuation.yield(.textDeltaPart(id: id, delta: text))
                            }
                            if let thinking = raw["delta"]?["message"]?["content"]?["thinking"]?.stringValue {
                                if activeReasoningIDs.insert(id).inserted {
                                    continuation.yield(.reasoningStart(id: id))
                                }
                                continuation.yield(.reasoningDeltaPart(id: id, delta: thinking))
                            }
                        case "content-end":
                            let id = String(raw["index"]?.intValue ?? 0)
                            if activeReasoningIDs.remove(id) != nil {
                                continuation.yield(.reasoningEnd(id: id))
                            } else if activeTextIDs.remove(id) != nil {
                                continuation.yield(.textEnd(id: id))
                            }
                        case "tool-call-start":
                            let toolCall = raw["delta"]?["message"]?["tool_calls"]
                            if let id = toolCall?["id"]?.stringValue,
                               let name = toolCall?["function"]?["name"]?.stringValue {
                                let arguments = toolCall?["function"]?["arguments"]?.stringValue ?? ""
                                pendingToolCall = CoherePendingToolCall(id: id, name: name, arguments: arguments, rawValue: toolCall)
                                continuation.yield(.toolInputStart(id: id, name: name))
                                continuation.yield(.toolCallDelta(id: id, name: name, argumentsDelta: arguments, index: nil))
                                if !arguments.isEmpty {
                                    continuation.yield(.toolInputDelta(id: id, delta: arguments))
                                }
                            }
                        case "tool-call-delta":
                            let arguments = raw["delta"]?["message"]?["tool_calls"]?["function"]?["arguments"]?.stringValue ?? ""
                            if var pending = pendingToolCall {
                                pending.arguments += arguments
                                pending.rawValue = raw["delta"]?["message"]?["tool_calls"] ?? pending.rawValue
                                pendingToolCall = pending
                                continuation.yield(.toolCallDelta(id: pending.id, name: pending.name, argumentsDelta: arguments, index: nil))
                                if !arguments.isEmpty {
                                    continuation.yield(.toolInputDelta(id: pending.id, delta: arguments))
                                }
                            }
                        case "tool-call-end":
                            if let pending = pendingToolCall {
                                continuation.yield(.toolInputEnd(id: pending.id))
                                continuation.yield(.toolCall(AIToolCall(
                                    id: pending.id,
                                    name: pending.name,
                                    arguments: cohereToolArguments(pending.arguments),
                                    rawValue: pending.rawValue
                                )))
                                pendingToolCall = nil
                            }
                        case "message-end":
                            finishReason = mapCohereFinishReason(raw["delta"]?["finish_reason"]?.stringValue)
                            usage = cohereTokenUsage(from: raw["delta"] ?? raw)
                            break eventLoop
                        default:
                            break
                        }
                    }
                    for id in activeTextIDs.sorted() {
                        continuation.yield(.textEnd(id: id))
                    }
                    for id in activeReasoningIDs.sorted() {
                        continuation.yield(.reasoningEnd(id: id))
                    }
                    continuation.yield(.finishMetadata(
                        reason: finishReason,
                        usage: usage,
                        providerMetadata: [:]
                    ))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func body(for request: LanguageModelRequest, stream: Bool) throws -> CoherePreparedCall {
        var warnings: [AIWarning] = []
        var options = try cohereProviderOptions(from: request)
        let responseFormat = cohereResolvedResponseFormat(request: request, options: &options)
        let toolChoice = request.toolChoice ?? options.removeValue(forKey: "toolChoice")
        let prompt = try coherePromptJSON(from: request.messages)
        var body: [String: JSONValue] = [
            "model": .string(modelID),
            "messages": .array(prompt.messages)
        ]
        if !prompt.documents.isEmpty { body["documents"] = .array(prompt.documents) }
        if stream { body["stream"] = true }
        if let maxOutputTokens = request.maxOutputTokens { body["max_tokens"] = .number(Double(maxOutputTokens)) }
        if let temperature = request.temperature { body["temperature"] = .number(temperature) }
        if let topP = request.topP { body["p"] = .number(topP) }
        if let topK = request.topK { body["k"] = .number(Double(topK)) }
        if let frequencyPenalty = request.frequencyPenalty { body["frequency_penalty"] = .number(frequencyPenalty) }
        if let presencePenalty = request.presencePenalty { body["presence_penalty"] = .number(presencePenalty) }
        if let seed = request.seed { body["seed"] = .number(Double(seed)) }
        if !request.stopSequences.isEmpty { body["stop_sequences"] = .array(request.stopSequences) }
        if let responseFormat {
            body["response_format"] = responseFormat
        }
        if let thinking = cohereThinking(reasoning: request.reasoning, options: &options, warnings: &warnings) {
            body["thinking"] = thinking
        }
        let preparedTools = cohereTools(from: request.tools, only: cohereForcedToolName(from: toolChoice))
        if !preparedTools.tools.isEmpty {
            body["tools"] = .array(preparedTools.tools)
            if let toolChoice = cohereToolChoice(from: toolChoice) {
                body["tool_choice"] = toolChoice
            }
        }
        for (key, value) in options where !["thinking", "responseFormat", "toolChoice"].contains(key) {
            body[key] = value
        }
        warnings.append(contentsOf: preparedTools.warnings)
        warnings.append(contentsOf: prompt.warnings)
        return CoherePreparedCall(body: body, warnings: warnings)
    }
}

struct CoherePreparedCall {
    var body: [String: JSONValue]
    var warnings: [AIWarning]
}

struct CoherePendingToolCall {
    var id: String
    var name: String
    var arguments: String
    var rawValue: JSONValue?
}

struct CoherePreparedTools {
    var tools: [JSONValue]
    var warnings: [AIWarning]
}

func cohereProviderOptions(from request: LanguageModelRequest) throws -> [String: JSONValue] {
    var output = cohereProviderOptions(from: request.extraBody)
    if let value = request.providerOptions["cohere"] {
        guard value != .null else { return output }
        guard let nested = value.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.cohere", message: "Cohere provider options must be an object.")
        }
        for key in cohereLanguageProviderOptionKeys {
            output.removeValue(forKey: key)
        }
        output.merge(try cohereValidateLanguageProviderOptions(nested)) { _, providerValue in providerValue }
    }
    return output
}

func cohereProviderOptions(from extraBody: [String: JSONValue]) -> [String: JSONValue] {
    var output = extraBody
    if let nested = output.removeValue(forKey: "cohere")?.objectValue {
        output.merge(nested) { _, nested in nested }
    }
    return output
}

let cohereLanguageProviderOptionKeys: Set<String> = ["thinking"]

func cohereResolvedResponseFormat(request: LanguageModelRequest, options: inout [String: JSONValue]) -> JSONValue? {
    if let responseFormat = request.responseFormat {
        options.removeValue(forKey: "responseFormat")
        return cohereResponseFormatJSON(responseFormat)
    }
    return cohereResponseFormat(from: options.removeValue(forKey: "responseFormat"))
}

func cohereResponseFormatJSON(_ responseFormat: AIResponseFormat) -> JSONValue? {
    switch responseFormat {
    case .text:
        return nil
    case let .json(schema, _, _):
        return .object([
            "type": .string("json_object"),
            "json_schema": schema
        ])
    }
}

func cohereResponseFormat(from value: JSONValue?) -> JSONValue? {
    guard let value else { return nil }
    if value["type"]?.stringValue == "json" {
        return .object([
            "type": .string("json_object"),
            "json_schema": value["schema"]
        ])
    }
    return value
}

func cohereThinking(reasoning: String?, options: inout [String: JSONValue], warnings: inout [AIWarning]) -> JSONValue? {
    if let thinking = options.removeValue(forKey: "thinking") {
        return cohereThinking(from: thinking)
    }
    guard let reasoning, isCustomReasoning(reasoning) else { return nil }
    if reasoning == "none" {
        return .object(["type": .string("disabled")])
    }
    guard let tokenBudget = mapReasoningToProviderBudget(
        reasoning: reasoning,
        maxOutputTokens: 32768,
        maxReasoningBudget: 32768,
        warnings: &warnings
    ) else {
        return nil
    }
    return .object([
        "type": .string("enabled"),
        "token_budget": .number(Double(tokenBudget))
    ])
}

func cohereThinking(from value: JSONValue) -> JSONValue {
    guard var object = value.objectValue else { return value }
    if let tokenBudget = object.removeValue(forKey: "tokenBudget") {
        object["token_budget"] = tokenBudget
    }
    if object["type"] == nil {
        object["type"] = .string("enabled")
    }
    return .object(object)
}

func cohereTools(from tools: [String: JSONValue], only forcedToolName: String? = nil) -> CoherePreparedTools {
    var warnings: [AIWarning] = []
    let values = tools.sorted { $0.key < $1.key }.compactMap { name, schema -> JSONValue? in
        let object = schema.objectValue
        if object?["type"]?.stringValue == "provider" || object?["id"]?.stringValue != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "provider-defined tool \(object?["id"]?.stringValue ?? name)"
            ))
            return nil
        }
        guard forcedToolName == nil || forcedToolName == name else { return nil }
        return .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(name),
                "description": schema["description"],
                "parameters": schema
            ])
        ])
    }
    return CoherePreparedTools(tools: values, warnings: warnings)
}

func cohereForcedToolName(from value: JSONValue?) -> String? {
    guard value?["type"]?.stringValue == "tool" else { return nil }
    return value?["toolName"]?.stringValue ?? value?["name"]?.stringValue
}

func cohereToolChoice(from value: JSONValue?) -> JSONValue? {
    switch value?["type"]?.stringValue ?? value?.stringValue {
    case "none":
        return .string("NONE")
    case "required", "tool":
        return .string("REQUIRED")
    case "auto", nil:
        return nil
    default:
        return value
    }
}

func coherePromptJSON(from messages: [AIMessage]) throws -> (messages: [JSONValue], documents: [JSONValue], warnings: [AIWarning]) {
    var cohereMessages: [JSONValue] = []
    var documents: [JSONValue] = []
    var warnings: [AIWarning] = []

    for message in messages {
        let converted = try cohereMessageJSON(message)
        cohereMessages.append(contentsOf: converted.messages)
        documents.append(contentsOf: converted.documents)
        warnings.append(contentsOf: converted.warnings)
    }

    return (cohereMessages, documents, warnings)
}

func cohereMessageJSON(_ message: AIMessage) throws -> (messages: [JSONValue], documents: [JSONValue], warnings: [AIWarning]) {
    switch message.role {
    case .system:
        return ([.object(["role": .string("system"), "content": .string(message.combinedText)])], [], [])
    case .assistant:
        var output: [String: JSONValue] = ["role": .string("assistant")]
        let toolCalls = message.content.compactMap(cohereAssistantToolCallJSON)
        if !toolCalls.isEmpty {
            output["tool_calls"] = .array(toolCalls)
        } else {
            output["content"] = .string(message.combinedText)
        }
        return ([.object(output)], [], [])
    case .tool:
        let toolResults = message.content.compactMap(cohereToolResultMessageJSON)
        if !toolResults.isEmpty {
            return (toolResults, [], [])
        }
        return ([.object(["role": .string("tool"), "content": .string(message.combinedText)])], [], [])
    case .user:
        let hasImage = message.content.contains {
            if let payload = $0.filePayload { return cohereIsImageMediaType(payload.mimeType) }
            if case .imageURL = $0 { return true }
            return false
        }
        var documents: [JSONValue] = []
        for part in message.content {
            if let document = try cohereDocumentJSON(part) {
                documents.append(document)
            }
        }
        guard hasImage else {
            let text = message.content.compactMap(\.text).joined()
            return ([.object(["role": .string("user"), "content": .string(text)])], documents, [])
        }
        var content: [JSONValue] = []
        for part in message.content {
            if let converted = try cohereContentPartJSON(part) {
                content.append(converted)
            }
        }
        return ([.object([
            "role": .string("user"),
            "content": .array(content)
        ])], documents, [])
    }
}

func cohereAssistantToolCallJSON(_ part: AIContentPart) -> JSONValue? {
    guard case let .toolCall(call) = part else { return nil }
    return .object([
        "id": .string(call.id),
        "type": .string("function"),
        "function": .object([
            "name": .string(call.name),
            "arguments": .string(call.arguments)
        ])
    ])
}

func cohereToolResultMessageJSON(_ part: AIContentPart) -> JSONValue? {
    guard case let .toolResult(result) = part else { return nil }
    return .object([
        "role": .string("tool"),
        "content": .string(cohereToolResultContent(result)),
        "tool_call_id": .string(result.toolCallID)
    ])
}

func cohereToolResultContent(_ result: AIToolResult) -> String {
    let output = result.modelOutput ?? result.result
    switch output["type"]?.stringValue {
    case "text", "error-text":
        return output["value"]?.stringValue ?? ""
    case "execution-denied":
        return output["reason"]?.stringValue ?? "Tool execution denied."
    case "content", "json", "error-json":
        if let value = output["value"] {
            return cohereJSONString(value) ?? value.stringValue ?? ""
        }
        return cohereJSONString(output) ?? output.stringValue ?? ""
    default:
        return cohereJSONString(output) ?? output.stringValue ?? ""
    }
}

func cohereJSONString(_ value: JSONValue) -> String? {
    guard let data = try? JSONEncoder().encode(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

func cohereContentPartJSON(_ part: AIContentPart) throws -> JSONValue? {
    switch part {
    case let .text(text, _):
        return text.isEmpty ? nil : .object(["type": .string("text"), "text": .string(text)])
    case let .reasoning(text, _):
        return text.isEmpty ? nil : .object(["type": .string("text"), "text": .string(text)])
    case let .imageURL(url, _):
        return .object(["type": .string("image_url"), "image_url": .object(["url": .string(url)])])
    case let .data(mimeType, data, _) where cohereIsImageMediaType(mimeType),
         let .file(mimeType, data, _, _) where cohereIsImageMediaType(mimeType):
        return .object([
            "type": .string("image_url"),
            "image_url": .object(["url": .string("data:\(cohereImageDataURLMediaType(mimeType));base64,\(data.base64EncodedString())")])
        ])
    case .providerReference:
        throw AIError.invalidArgument(
            argument: "files",
            message: "Cohere chat API expects file URLs to be downloaded before prompt conversion."
        )
    case .data, .file, .reasoningFile, .custom, .toolCall, .toolResult, .toolApprovalRequest, .toolApprovalResponse:
        return nil
    }
}

func cohereDocumentJSON(_ part: AIContentPart) throws -> JSONValue? {
    guard let payload = part.filePayload, !cohereIsImageMediaType(payload.mimeType) else {
        return nil
    }
    guard payload.mimeType.hasPrefix("text/") || payload.mimeType == "application/json" else {
        throw AIError.invalidArgument(
            argument: "files",
            message: "Media type '\(payload.mimeType)' is not supported. Supported media types are: text/* and application/json."
        )
    }
    var data: [String: JSONValue] = [
        "text": .string(String(decoding: payload.data, as: UTF8.self))
    ]
    if let filename = payload.filename { data["title"] = .string(filename) }
    return .object(["data": .object(data)])
}

func cohereIsImageMediaType(_ mediaType: String) -> Bool {
    mediaType == "image" || mediaType.hasPrefix("image/")
}

func cohereImageDataURLMediaType(_ mediaType: String) -> String {
    mediaType == "image" || mediaType == "image/*" ? "image/jpeg" : mediaType
}

func mapCohereFinishReason(_ reason: String?) -> String? {
    switch reason {
    case "COMPLETE", "STOP_SEQUENCE":
        return "stop"
    case "MAX_TOKENS":
        return "length"
    case "ERROR":
        return "error"
    case "TOOL_CALL":
        return "tool-calls"
    default:
        return "other"
    }
}

func cohereToolCalls(from value: JSONValue?) -> [AIToolCall] {
    value?.arrayValue?.enumerated().compactMap { index, item in
        guard let name = item["function"]?["name"]?.stringValue else { return nil }
        return AIToolCall(
            id: item["id"]?.stringValue ?? "tool-call-\(index)",
            name: name,
            arguments: cohereToolArguments(item["function"]?["arguments"]?.stringValue ?? ""),
            rawValue: item
        )
    } ?? []
}

func cohereSources(from value: JSONValue?) -> [AISource] {
    value?.arrayValue?.enumerated().map { index, citation in
        let document = citation["sources"]?[0]?["document"]
        var metadata: [String: JSONValue] = [:]
        if let start = citation["start"] { metadata["start"] = start }
        if let end = citation["end"] { metadata["end"] = end }
        if let text = citation["text"] { metadata["text"] = text }
        if let sources = citation["sources"] { metadata["sources"] = sources }
        if let citationType = citation["type"] { metadata["citationType"] = citationType }
        return AISource(
            id: "cohere-citation-\(index)",
            sourceType: "document",
            title: document?["title"]?.stringValue ?? "Document",
            mediaType: "text/plain",
            providerMetadata: ["cohere": .object(metadata)],
            rawValue: citation
        )
    } ?? []
}

func cohereToolArguments(_ arguments: String) -> String {
    let trimmed = arguments.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty || trimmed == "null" {
        return "{}"
    }
    if let json = try? decodeJSONBody(Data(trimmed.utf8)),
       let canonical = cohereJSONString(json) {
        return canonical
    }
    return trimmed
}

func cohereTokenUsage(from raw: JSONValue) -> TokenUsage? {
    guard let usage = raw["usage"], let tokens = usage["tokens"] else { return nil }
    let inputTokens = tokens["input_tokens"]?.intValue
    let cachedTokens = usage["cached_tokens"]?.intValue
    return TokenUsage(
        inputTokens: inputTokens,
        outputTokens: tokens["output_tokens"]?.intValue,
        totalTokens: (tokens["input_tokens"]?.intValue).flatMap { input in
            (tokens["output_tokens"]?.intValue).map { input + $0 }
        },
        inputTokensNoCache: inputTokens.map { $0 - (cachedTokens ?? 0) },
        inputTokensCacheRead: cachedTokens,
        outputTextTokens: tokens["output_tokens"]?.intValue,
        rawValue: usage
    )
}

func cohereResponseMetadata(from raw: JSONValue? = nil, response: AIHTTPResponse, modelID: String) -> AIResponseMetadata {
    AIResponseMetadata(
        id: raw?["id"]?.stringValue ?? raw?["generation_id"]?.stringValue,
        timestamp: Date(),
        modelID: raw?["model"]?.stringValue ?? modelID,
        headers: response.headers,
        body: raw
    )
}