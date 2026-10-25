@testable import SwiftAICohere
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func cohereLanguageProviderOptionsFollowUpstreamSchema() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"message":{"role":"assistant","content":[{"type":"text","text":"ok"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}"#))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-reasoning-08-2025")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        reasoning: "none",
        providerOptions: [
            "cohere": [
                "thinking": [
                    "type": "disabled",
                    "tokenBudget": 256,
                    "extra": "drop-me"
                ],
                "safetyMode": "drop-me"
            ]
        ],
        extraBody: [
            "cohere": [
                "thinking": [
                    "type": "enabled",
                    "tokenBudget": 1024
                ]
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["cohere"] == nil)
    #expect(body["safetyMode"] == nil)
    #expect(body["thinking"]?["type"]?.stringValue == "disabled")
    #expect(body["thinking"]?["token_budget"]?.intValue == 256)
    #expect(body["thinking"]?["extra"] == nil)
}

@Test func cohereLanguageProviderOptionsValidateLikeUpstreamSchema() async throws {
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: RecordingTransport(response: jsonResponse("{}"))))
    let model = try provider.languageModel("command-a-reasoning-08-2025")

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere", message: "Cohere provider options must be an object.")) {
        _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], providerOptions: ["cohere": "bad"]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.thinking", message: "Cohere thinking cannot be null.")) {
        _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], providerOptions: ["cohere": ["thinking": .null]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.thinking.type", message: "Cohere thinking.type must be enabled or disabled.")) {
        _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], providerOptions: ["cohere": ["thinking": ["type": "auto"]]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.thinking.tokenBudget", message: "Cohere thinking.tokenBudget must be a number.")) {
        _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], providerOptions: ["cohere": ["thinking": ["tokenBudget": "256"]]]))
    }
}

@Test func cohereEmbeddingProviderOptionsValidateLikeUpstreamSchema() async throws {
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: RecordingTransport(response: jsonResponse(#"{"embeddings":{"float":[[0.1]]},"meta":{"billed_units":{"input_tokens":1}}}"#))))
    let model = try provider.embeddingModel("embed-v4.0")

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere", message: "Cohere provider options must be an object.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["hello"], providerOptions: ["cohere": true]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.inputType", message: "Cohere inputType cannot be null.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["hello"], providerOptions: ["cohere": ["inputType": .null]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.inputType", message: "Cohere inputType must be one of search_document, search_query, classification, clustering.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["hello"], providerOptions: ["cohere": ["inputType": "query"]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.truncate", message: "Cohere truncate must be one of NONE, START, END.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["hello"], providerOptions: ["cohere": ["truncate": "MIDDLE"]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.outputDimension", message: "Cohere outputDimension must be one of 256, 512, 768, 1024, 1536, 2048.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["hello"], providerOptions: ["cohere": ["outputDimension": 769]]))
    }
}

@Test func cohereRerankingProviderOptionsValidateLikeUpstreamSchema() async throws {
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: RecordingTransport(response: jsonResponse(#"{"results":[{"index":0,"relevance_score":0.8}]}"#))))
    let model = try provider.rerankingModel("rerank-v3.5")

    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere", message: "Cohere provider options must be an object.")) {
        _ = try await model.rerank(RerankingRequest(query: "q", documents: ["a"], providerOptions: ["cohere": false]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.maxTokensPerDoc", message: "Cohere maxTokensPerDoc cannot be null.")) {
        _ = try await model.rerank(RerankingRequest(query: "q", documents: ["a"], providerOptions: ["cohere": ["maxTokensPerDoc": .null]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.cohere.priority", message: "Cohere priority must be a number.")) {
        _ = try await model.rerank(RerankingRequest(query: "q", documents: ["a"], providerOptions: ["cohere": ["priority": "1"]]))
    }
}
