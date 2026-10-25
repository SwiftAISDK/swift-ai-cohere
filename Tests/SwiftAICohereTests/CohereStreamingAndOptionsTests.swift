@testable import SwiftAICohere
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import Testing
import SwiftAI

@Test func cohereLanguageStreamsChatEvents() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"type":"message-start","id":"msg-1"}

    data: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text"}}}}

    data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"co"}}}}

    data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"here"}}}}

    data: {"type":"content-end","index":0}

    data: {"type":"message-end","delta":{"finish_reason":"MAX_TOKENS","usage":{"tokens":{"input_tokens":1,"output_tokens":2}}}}

    """, headers: ["x-stream": "yes"]))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    var deltas: [String] = []
    var lifecycle: [String] = []
    var finishReason: String?
    var streamStartWarnings: [AIWarning]?
    var responseMetadata: AIResponseMetadata?
    for try await part in model.stream(LanguageModelRequest(
        messages: [.user("Hi")],
        topK: 5,
        providerOptions: ["cohere": ["priority": 1]]
    )) {
        switch part {
        case let .textDeltaPart(id, delta, _):
            lifecycle.append("delta:\(id):\(delta)")
            deltas.append(delta)
        case let .textStart(id, _):
            lifecycle.append("start:\(id)")
        case let .textEnd(id, _), let .textEndWithCitations(id, _, _):
            lifecycle.append("end:\(id)")
        case let .streamStart(warnings):
            streamStartWarnings = warnings
        case let .responseMetadata(metadata):
            if metadata.id == "msg-1" {
                responseMetadata = metadata
            }
        case let .finishMetadata(reason, _, _):
            finishReason = reason
        default:
            break
        }
    }

    #expect(deltas == ["co", "here"])
    #expect(lifecycle == ["start:0", "delta:0:co", "delta:0:here", "end:0"])
    #expect(finishReason == "length")
    #expect(streamStartWarnings == [])
    #expect(responseMetadata?.id == "msg-1")
    #expect(responseMetadata?.headers["x-stream"] == "yes")
    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["stream"] == true)
    #expect(body["k"]?.intValue == 5)
    #expect(body["priority"] == nil)
}
@Test func cohereLanguageStreamsToolCallEvents() async throws {
    let transport = RecordingTransport(response: sseResponse(#"""
    data: {"type":"tool-call-start","delta":{"message":{"tool_calls":{"id":"weather_dqgshstja6p9","type":"function","function":{"name":"weather","arguments":"{ \"location\" :"}}}}}

    data: {"type":"tool-call-delta","delta":{"message":{"tool_calls":{"function":{"arguments":" \"San Francisco\" }"}}}}}

    data: {"type":"tool-call-end"}

    data: {"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":3,"output_tokens":2}}}}

    """#))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    var deltas: [String] = []
    var inputLifecycle: [String] = []
    var finalCall: AIToolCall?
    var finishReason: String?
    var totalTokens: Int?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Weather?")])) {
        switch part {
        case let .toolInputStart(id, name, _, _, _, _):
            inputLifecycle.append("start:\(id):\(name)")
        case let .toolInputDelta(id, delta, _):
            inputLifecycle.append("delta:\(id):\(delta)")
        case let .toolInputEnd(id, _):
            inputLifecycle.append("end:\(id)")
        case let .toolCallDelta(_, _, argumentsDelta, _):
            deltas.append(argumentsDelta)
        case let .toolCall(call):
            finalCall = call
        case let .finishMetadata(reason, usage, _):
            finishReason = reason
            totalTokens = usage?.totalTokens
        default:
            break
        }
    }

    let call = try #require(finalCall)
    #expect(deltas == ["{ \"location\" :", " \"San Francisco\" }"])
    #expect(inputLifecycle == [
        "start:weather_dqgshstja6p9:weather",
        "delta:weather_dqgshstja6p9:{ \"location\" :",
        "delta:weather_dqgshstja6p9: \"San Francisco\" }",
        "end:weather_dqgshstja6p9"
    ])
    #expect(call.id == "weather_dqgshstja6p9")
    #expect(call.name == "weather")
    #expect(call.arguments == #"{"location":"San Francisco"}"#)
    #expect(try decodeJSONBody(Data(call.arguments.utf8))["location"]?.stringValue == "San Francisco")
    #expect(finishReason == "tool-calls")
    #expect(totalTokens == 5)
}
@Test func cohereLanguageStreamsParseErrorsAsErrorPartsLikeUpstream() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"type":"message-start","id":"msg-err"}

    data: not-json

    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    var errors: [String] = []
    var finishReason: String?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
        switch part {
        case let .error(message, _):
            errors.append(message)
        case let .finishMetadata(reason, _, _):
            finishReason = reason
        default:
            break
        }
    }

    #expect(errors.count == 1)
    #expect(finishReason == "error")
}
@Test func cohereEmbeddingAndRerankingUseNativeEndpoints() async throws {
    let embeddingTransport = RecordingTransport(response: jsonResponse("""
    {"embeddings":{"float":[[0.1,0.2],[0.3,0.4]]},"meta":{"billed_units":{"input_tokens":7}}}
    """))
    let embeddingProvider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: embeddingTransport))
    let embeddingModel = try embeddingProvider.embeddingModel("embed-english-v3.0")
    #expect(embeddingModel.maxEmbeddingsPerCall == 96)

    let embedding = try await embeddingModel.embed(EmbeddingRequest(values: ["hello", "world"], dimensions: 512, extraBody: ["inputType": "classification", "truncate": "END"]))

    #expect(embedding.embeddings == [[0.1, 0.2], [0.3, 0.4]])
    #expect(embedding.usage?.totalTokens == 7)
    let embeddingRequest = try #require(await embeddingTransport.requests().first)
    #expect(embeddingRequest.url.absoluteString == "https://api.cohere.com/v2/embed")
    let embeddingBody = try decodeJSONBody(try #require(embeddingRequest.body))
    #expect(embeddingBody["texts"]?[0]?.stringValue == "hello")
    #expect(embeddingBody["embedding_types"]?[0]?.stringValue == "float")
    #expect(embeddingBody["input_type"]?.stringValue == "classification")
    #expect(embeddingBody["output_dimension"]?.intValue == 512)
    #expect(embeddingBody["truncate"]?.stringValue == "END")

    let tooManyValues = Array(repeating: "x", count: 97)
    await #expect(throws: AITooManyEmbeddingValuesForCallError(
        provider: "cohere.textEmbedding",
        modelID: "embed-english-v3.0",
        maxEmbeddingsPerCall: 96,
        values: tooManyValues
    )) {
        _ = try await embeddingModel.embed(EmbeddingRequest(values: tooManyValues))
    }
    #expect(await embeddingTransport.requests().count == 1)

    let rerankTransport = RecordingTransport(response: jsonResponse(#"{"id":"rank-1","results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#))
    let rerankProvider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: rerankTransport))
    let rerankModel = try rerankProvider.rerankingModel("rerank-v3.5")

    let reranking = try await rerankModel.rerank(RerankingRequest(query: "q", documents: ["a", "b"], topK: 1, extraBody: ["maxTokensPerDoc": 256]))

    #expect(reranking.results.map(\.index) == [1, 0])
    let rerankRequest = try #require(await rerankTransport.requests().first)
    #expect(rerankRequest.url.absoluteString == "https://api.cohere.com/v2/rerank")
    let rerankBody = try decodeJSONBody(try #require(rerankRequest.body))
    #expect(rerankBody["top_n"]?.intValue == 1)
    #expect(rerankBody["max_tokens_per_doc"]?.intValue == 256)
}
@Test func cohereModelsMapNestedProviderOptions() async throws {
    let chatTransport = RecordingTransport(response: jsonResponse(#"{"message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}"#))
    let chatProvider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: chatTransport))
    let chatModel = try chatProvider.languageModel("command-a-reasoning-08-2025")

    _ = try await chatModel.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        extraBody: [
            "cohere": [
                "thinking": [
                    "type": "enabled",
                    "tokenBudget": 128
                ]
            ]
        ]
    ))

    let chatRequest = try #require(await chatTransport.requests().first)
    let chatBody = try decodeJSONBody(try #require(chatRequest.body))
    #expect(chatBody["cohere"] == nil)
    #expect(chatBody["thinking"]?["type"]?.stringValue == "enabled")
    #expect(chatBody["thinking"]?["token_budget"]?.intValue == 128)

    let embeddingTransport = RecordingTransport(response: jsonResponse(#"{"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":3}}}"#))
    let embeddingProvider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: embeddingTransport))
    let embeddingModel = try embeddingProvider.embeddingModel("embed-v4.0")

    _ = try await embeddingModel.embed(EmbeddingRequest(
        values: ["hello"],
        providerOptions: [
            "cohere": [
                "inputType": "clustering",
                "outputDimension": 1536,
                "truncate": "NONE",
                "priority": 999
            ],
            "voyage": [
                "inputType": "query"
            ]
        ],
        extraBody: [
            "cohere": [
                "inputType": "search_document",
                "outputDimension": 1024,
                "truncate": "START",
                "rawExtra": "keep-me"
            ]
        ]
    ))

    let embeddingRequest = try #require(await embeddingTransport.requests().first)
    let embeddingBody = try decodeJSONBody(try #require(embeddingRequest.body))
    #expect(embeddingBody["cohere"] == nil)
    #expect(embeddingBody["voyage"] == nil)
    #expect(embeddingBody["priority"] == nil)
    #expect(embeddingBody["input_type"]?.stringValue == "clustering")
    #expect(embeddingBody["output_dimension"]?.intValue == 1536)
    #expect(embeddingBody["truncate"]?.stringValue == "NONE")
    #expect(embeddingBody["rawExtra"]?.stringValue == "keep-me")

    let rerankTransport = RecordingTransport(response: jsonResponse(#"{"results":[{"index":0,"relevance_score":0.8}]}"#))
    let rerankProvider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: rerankTransport))
    let rerankModel = try rerankProvider.rerankingModel("rerank-v3.5")

    let rerankResult = try await rerankModel.rerank(RerankingRequest(
        query: "q",
        documents: [["body": "a", "source": "doc-1"]],
        providerOptions: [
            "cohere": [
                "maxTokensPerDoc": 512,
                "priority": 2,
                "truncate": "drop-me"
            ],
            "voyage": [
                "returnDocuments": true
            ]
        ],
        extraBody: [
            "cohere": [
                "maxTokensPerDoc": 128,
                "priority": 1,
                "rawRerank": true
            ]
        ]
    ))

    #expect(rerankResult.warnings == [
        AIWarning(type: "compatibility", feature: "object documents", message: "Object documents are converted to strings.")
    ])
    let rerankRequest = try #require(await rerankTransport.requests().first)
    let rerankBody = try decodeJSONBody(try #require(rerankRequest.body))
    #expect(rerankBody["cohere"] == nil)
    #expect(rerankBody["voyage"] == nil)
    #expect(rerankBody["truncate"] == nil)
    #expect(rerankBody["max_tokens_per_doc"]?.intValue == 512)
    #expect(rerankBody["priority"]?.intValue == 2)
    #expect(rerankBody["rawRerank"]?.boolValue == true)
    let documentText = try #require(rerankBody["documents"]?[0]?.stringValue)
    let documentJSON = try decodeJSONBody(Data(documentText.utf8))
    #expect(documentJSON["body"]?.stringValue == "a")
    #expect(documentJSON["source"]?.stringValue == "doc-1")
}