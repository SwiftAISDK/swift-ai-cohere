@testable import SwiftAICohere
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import Testing

// Translations of @ai-sdk/cohere 4.0.84 tests (cohere-chat-language-model.test.ts,
// convert-to-cohere-chat-prompt.test.ts, cohere-prepare-tools.test.ts) that cover
// observable request/stream behavior.

private let cohereTextResponse = """
{"id":"e7592632-1e3d-424f-b129-bd5f9f980f7b","message":{"role":"assistant","content":[{"type":"text","text":"Hello"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":4,"output_tokens":1}}}
"""

private func cohereModel(_ transport: RecordingTransport) throws -> any LanguageModel {
    try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "test-api-key", transport: transport))
        .languageModel("command-r-plus")
}

private func cohereRequestBody(_ transport: RecordingTransport) async throws -> JSONValue {
    try decodeJSONBody(try #require(await transport.requests().first?.body))
}

private func cohereChunkLines(_ chunks: [String]) -> AIHTTPResponse {
    sseResponse(chunks.map { "data: \($0)\n\n" }.joined())
}

private func collect(_ stream: AsyncThrowingStream<LanguageStreamPart, Error>) async throws -> [LanguageStreamPart] {
    var parts: [LanguageStreamPart] = []
    for try await part in stream { parts.append(part) }
    return parts
}

private let currentTimeTool: [String: JSONValue] = [
    "currentTime": ["type": "object", "properties": [:], "additionalProperties": false]
]

// MARK: - convert-to-cohere-chat-prompt.test.ts

@Test func cohereUpstreamAcceptsTopLevelOnlyTextMediaType() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .file(mimeType: "text", data: Data("This is file content".utf8), filename: "test.txt")
        ])
    ]))
    let body = try await cohereRequestBody(transport)
    #expect(body["messages"] == [["role": "user", "content": ""]])
    #expect(body["documents"] == [["data": ["text": "This is file content", "title": "test.txt"]]])
}

@Test func cohereUpstreamPassesImageDetailProviderOption() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .imageURL("https://example.com/image.png", providerMetadata: ["cohere": ["detail": "high"]])
        ])
    ]))
    let body = try await cohereRequestBody(transport)
    #expect(body["messages"] == [[
        "role": "user",
        "content": [["type": "image_url", "image_url": ["url": "https://example.com/image.png", "detail": "high"]]]
    ]])
}

@Test func cohereUpstreamOmitsImageDetailWithoutProviderOption() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .file(mimeType: "image/png", data: Data([1, 2, 3]))
        ])
    ]))
    let body = try await cohereRequestBody(transport)
    #expect(body["messages"] == [[
        "role": "user",
        "content": [["type": "image_url", "image_url": ["url": "data:image/png;base64,AQID"]]]
    ]])
}

@Test func cohereUpstreamDetectsFullImageMediaTypeFromBytes() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .data(mimeType: "image", data: Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))
        ])
    ]))
    let body = try await cohereRequestBody(transport)
    let url = body["messages"]?[0]?["content"]?[0]?["image_url"]?["url"]?.stringValue
    #expect(body["messages"]?[0]?["content"]?[0]?["type"] == "image_url")
    #expect(url?.hasPrefix("data:image/png;base64,") == true)
}

@Test func cohereUpstreamConvertsMultipleToolResultsAndSkipsApprovalResponses() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .tool, content: [
            .toolApprovalResponse(AIToolApprovalResponse(id: "approval-1", approved: true)),
            .toolResult(AIToolResult(toolCallID: "tool-call-1", toolName: "tool-1", result: ["test": "This is a tool message"], modelOutput: ["type": "json", "value": ["test": "This is a tool message"]])),
            .toolResult(AIToolResult(toolCallID: "tool-call-2", toolName: "tool-2", result: ["something": "else"], modelOutput: ["type": "json", "value": ["something": "else"]]))
        ]),
        AIMessage(role: .tool, content: [
            .toolApprovalResponse(AIToolApprovalResponse(id: "approval-2", approved: false))
        ])
    ]))
    let body = try await cohereRequestBody(transport)
    #expect(body["messages"] == [
        ["role": "tool", "content": #"{"test":"This is a tool message"}"#, "tool_call_id": "tool-call-1"],
        ["role": "tool", "content": #"{"something":"else"}"#, "tool_call_id": "tool-call-2"]
    ])
}

@Test func cohereUpstreamConcatenatesAssistantTextWithoutSeparator() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .assistant, content: [.text("Hello"), .text(" world")])
    ]))
    let body = try await cohereRequestBody(transport)
    #expect(body["messages"] == [["role": "assistant", "content": "Hello world"]])
}

@Test(arguments: [#"{"query":"#, "unparseable"])
func cohereUpstreamStringifiesInvalidJSONToolCallArguments(arguments: String) async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [
        AIMessage(role: .assistant, content: [.toolCall(AIToolCall(id: "call-1", name: "search", arguments: arguments))])
    ]))
    let body = try await cohereRequestBody(transport)
    let serialized = try #require(body["messages"]?[0]?["tool_calls"]?[0]?["function"]?["arguments"]?.stringValue)
    #expect(try decodeJSONBody(Data(serialized.utf8)) == .string(arguments))
}

// MARK: - cohere-chat-language-model.test.ts (doGenerate)

@Test func cohereUpstreamGenerateHandlesStringNullToolCallArguments() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"id-1","message":{"role":"assistant","content":[],"tool_plan":"plan","tool_calls":[{"id":"currentTime_tf4dywn8wgnk","type":"function","function":{"name":"currentTime","arguments":"null"}}]},"finish_reason":"TOOL_CALL"}
    """))
    let result = try await cohereModel(transport).generate(LanguageModelRequest(messages: [.user("Hello")], tools: currentTimeTool))
    #expect(result.toolCalls.map(\.arguments) == ["{}"])
    #expect(result.toolCalls.map(\.id) == ["currentTime_tf4dywn8wgnk"])
}

@Test func cohereUpstreamGenerateKeepsToolCallArgumentsVerbatim() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"id-1","message":{"role":"assistant","content":[],"tool_calls":[{"id":"call-1","type":"function","function":{"name":"fetch","arguments":"{\\"url\\": \\"https://example.com/a\\", \\"a\\": 1}"}}]},"finish_reason":"TOOL_CALL"}
    """))
    let result = try await cohereModel(transport).generate(LanguageModelRequest(messages: [.user("Hello")]))
    #expect(result.toolCalls.map(\.arguments) == [#"{"url": "https://example.com/a", "a": 1}"#])
}

@Test func cohereUpstreamGenerateValidatesCachedTokenUsage() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"message":{"role":"assistant","content":[{"type":"text","text":"Hello"}]},"finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":12,"output_tokens":7},"tokens":{"input_tokens":507,"output_tokens":10},"cached_tokens":"448"}}
    """))
    await #expect(throws: (any Error).self) {
        _ = try await cohereModel(transport).generate(LanguageModelRequest(messages: [.user("Hello")]))
    }
}

@Test func cohereUpstreamSendsEmptyToolsWhenOnlyProviderToolsArePresent() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    let result = try await cohereModel(transport).generate(LanguageModelRequest(
        messages: [.user("Hello")],
        tools: ["web": ["type": "provider", "id": "openai.web_search", "args": [:]]]
    ))
    let body = try await cohereRequestBody(transport)
    #expect(body["tools"] == [])
    #expect(body["tool_choice"] == nil)
    #expect(result.warnings.map(\.feature) == ["provider-defined tool openai.web_search"])
}

@Test func cohereUpstreamFiltersToolsForSpecificToolChoice() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    _ = try await cohereModel(transport).generate(LanguageModelRequest(
        messages: [.user("Hello")],
        tools: [
            "tool1": ["type": "object", "properties": [:]],
            "tool2": ["type": "object", "properties": [:]]
        ],
        toolChoice: ["type": "tool", "toolName": "tool2"]
    ))
    let body = try await cohereRequestBody(transport)
    #expect(body["tools"]?.arrayValue?.map { $0["function"]?["name"] } == ["tool2"])
    #expect(body["tool_choice"] == "REQUIRED")
}

@Test func cohereUpstreamOrdersToolWarningsBeforeReasoningWarnings() async throws {
    let transport = RecordingTransport(response: jsonResponse(cohereTextResponse))
    let result = try await cohereModel(transport).generate(LanguageModelRequest(
        messages: [.user("Hello")],
        reasoning: "xhigh",
        tools: ["web": ["type": "provider", "id": "openai.web_search", "args": [:]]]
    ))
    #expect(result.warnings.first?.feature == "provider-defined tool openai.web_search")
    #expect(result.warnings.count >= 1)
}

// MARK: - cohere-chat-language-model.test.ts (doStream)

@Test func cohereUpstreamStreamHandlesStringNullToolCallArguments() async throws {
    let transport = RecordingTransport(response: cohereChunkLines([
        #"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"test-tool-call","type":"function","function":{"name":"currentTime","arguments":""}}}}}"#,
        #"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"null"}}}}}"#,
        #"{"type":"tool-call-end","index":0}"#
    ]))
    let parts = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")], tools: currentTimeTool)))
    let calls = parts.compactMap { part -> AIToolCall? in
        if case let .toolCall(call) = part { return call }
        return nil
    }
    #expect(calls.map(\.id) == ["test-tool-call"])
    #expect(calls.map(\.name) == ["currentTime"])
    #expect(calls.map(\.arguments) == ["{}"])
}

@Test func cohereUpstreamStreamHandlesEmptyToolCallArguments() async throws {
    let transport = RecordingTransport(response: cohereChunkLines([
        #"{"type":"message-start","id":"msg-1"}"#,
        #"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"test-id","type":"function","function":{"name":"test-tool","arguments":""}}}}}"#,
        #"{"type":"tool-call-end","index":0}"#,
        #"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}}"#
    ]))
    let parts = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")])))
    let calls = parts.compactMap { part -> AIToolCall? in
        if case let .toolCall(call) = part { return call }
        return nil
    }
    #expect(calls.map(\.arguments) == ["{}"])
    #expect(!parts.contains { if case .toolInputDelta = $0 { true } else { false } })
}

@Test func cohereUpstreamStreamRejectsPrototypeKeysInToolCallArguments() async throws {
    let transport = RecordingTransport(response: cohereChunkLines([
        #"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"test-tool-call","type":"function","function":{"name":"test-tool","arguments":""}}}}}"#,
        #"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"{\"__proto__\":{\"polluted\":true}}"}}}}}"#,
        #"{"type":"tool-call-end","index":0}"#
    ]))
    await #expect(throws: (any Error).self) {
        _ = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")])))
    }
}

@Test func cohereUpstreamStreamHandlesUnparsableStreamParts() async throws {
    let transport = RecordingTransport(response: sseResponse("event: foo-message\ndata: {unparsable}\n\n"))
    let parts = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")])))
    let errors = parts.filter { if case .error = $0 { true } else { false } }
    #expect(errors.count == 1)
    guard case let .finishMetadata(reason, usage, _)? = parts.last else {
        Issue.record("missing finish")
        return
    }
    #expect(reason == "error")
    #expect(usage == nil)
}

@Test(arguments: [false, true])
func cohereUpstreamStreamHandlesTruncatedStream(withError: Bool) async throws {
    var chunks = [
        #"{"type":"message-start","id":"29f14a5a-11de-4cae-9800-25e4747408ea"}"#,
        #"{"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}"#,
        #"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hello"}}}}"#
    ]
    if withError { chunks.append("{invalid-json}") }
    let transport = RecordingTransport(response: cohereChunkLines(chunks))
    let parts = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")])))
    #expect(parts.contains { if case .textDeltaPart = $0 { true } else { false } })
    #expect(parts.filter { if case .error = $0 { true } else { false } }.count == (withError ? 1 : 0))
    let finishes = parts.compactMap { part -> String?? in
        if case let .finishMetadata(reason, _, _) = part { return .some(reason) }
        return nil
    }
    #expect(finishes == [withError ? "error" : "interrupted"])
}

@Test(arguments: [
    #"{"billed_units":{"input_tokens":"12","output_tokens":7},"tokens":{"input_tokens":507,"output_tokens":10},"cached_tokens":448}"#,
    #"{"billed_units":{"input_tokens":12,"output_tokens":7},"tokens":{"input_tokens":507,"output_tokens":10},"cached_tokens":"448"}"#
])
func cohereUpstreamStreamValidatesUsage(usage: String) async throws {
    let transport = RecordingTransport(response: cohereChunkLines([
        #"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":"# + usage + "}}"
    ]))
    let parts = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")])))
    let kinds = parts.compactMap { part -> String? in
        switch part {
        case .streamStart: "stream-start"
        case .error: "error"
        case let .finishMetadata(reason, usage, _): "finish:\(reason ?? "nil"):\(usage == nil)"
        default: nil
        }
    }
    #expect(kinds == ["stream-start", "error", "finish:error:true"])
}

@Test func cohereUpstreamStreamTreatsUnknownEventTypesAsValidationErrors() async throws {
    let transport = RecordingTransport(response: cohereChunkLines([
        #"{"type":"citation-end"}"#,
        #"{"type":"tool-plan-delta","delta":{"message":{"tool_plan":"plan"}}}"#,
        #"{"type":"unknown-event"}"#
    ]))
    let parts = try await collect(cohereModel(transport).stream(LanguageModelRequest(messages: [.user("Hello")])))
    #expect(parts.filter { if case .error = $0 { true } else { false } }.count == 1)
    guard case let .finishMetadata(reason, _, _)? = parts.last else {
        Issue.record("missing finish")
        return
    }
    #expect(reason == "error")
}
