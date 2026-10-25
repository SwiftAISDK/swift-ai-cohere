@testable import SwiftAICohere
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import Testing
import SwiftAI


@Test func cohereLanguageUsesChatEndpointAndCohereShape() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"co"},{"type":"text","text":"here"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":3,"output_tokens":2},"billed_units":{"input_tokens":3,"output_tokens":2}}}
    """, headers: ["x-cohere": "yes"]))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    let result = try await model.generate(LanguageModelRequest(messages: [.system("Brief."), .user("Hi")], topP: 0.8, maxOutputTokens: 12))

    #expect(result.text == "cohere")
    #expect(result.finishReason == "stop")
    #expect(result.usage?.totalTokens == 5)
    #expect(result.responseMetadata.id == "gen-1")
    #expect(result.responseMetadata.modelID == "command-a-03-2025")
    #expect(result.responseMetadata.headers["x-cohere"] == "yes")
    #expect(result.responseMetadata.body?["generation_id"]?.stringValue == "gen-1")
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://api.cohere.com/v2/chat")
    #expect(request.headers["authorization"] == "Bearer cohere-key")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["model"]?.stringValue == "command-a-03-2025")
    #expect(body["messages"]?[0]?["role"]?.stringValue == "system")
    #expect(body["messages"]?[1]?["content"]?.stringValue == "Hi")
    #expect(body["p"]?.doubleValue == 0.8)
    #expect(body["max_tokens"]?.intValue == 12)
    #expect(body["documents"] == nil)
}

@Test func cohereLanguagePreservesCompleteRawUsageLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"generation_id":"gen-raw","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":5,"future_nested":{"kept":true}},"billed_units":{"input_tokens":9,"output_tokens":4},"cached_tokens":3,"future_top_level":{"label":"kept"}}}"#))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))

    let result = try await provider.languageModel("command-a-03-2025").generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.usage?.inputTokens == 10)
    #expect(result.usage?.outputTokens == 5)
    #expect(result.usage?.rawValue?["billed_units"]?["input_tokens"]?.intValue == 9)
    #expect(result.usage?.rawValue?["cached_tokens"]?.intValue == 3)
    #expect(result.usage?.rawValue?["tokens"]?["future_nested"]?["kept"]?.boolValue == true)
    #expect(result.usage?.rawValue?["future_top_level"]?["label"]?.stringValue == "kept")
}

@Test func cohereProviderAddsVersionedUserAgentSuffix() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(
        apiKey: "cohere-key",
        headers: ["User-Agent": "custom-client/1.0"],
        transport: transport
    ))
    let model = try provider.languageModel("command-a-03-2025")

    _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    let request = try #require(await transport.requests().first)
    #expect(request.headers["authorization"] == "Bearer cohere-key")
    #expect(request.headers["user-agent"] == "custom-client/1.0 ai-sdk-cohere/4.0.69")
}

@Test func cohereLanguageMapsTopLevelReasoningLikeUpstreamV4() async throws {
    let response = jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}
    """)
    let transport = RecordingTransport(responses: [response, response, response, response, response])
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-reasoning-08-2025")

    _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], reasoning: "high"))
    _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], reasoning: "none"))
    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        reasoning: "none",
        providerOptions: ["cohere": ["thinking": ["type": "enabled"]]]
    ))
    _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], reasoning: "provider-default"))
    let unsupported = try await model.generate(LanguageModelRequest(messages: [.user("Hi")], reasoning: "unsupported-level"))

    let requests = await transport.requests()
    let highBody = try decodeJSONBody(try #require(requests[0].body))
    #expect(highBody["thinking"]?["type"]?.stringValue == "enabled")
    #expect(highBody["thinking"]?["token_budget"]?.intValue == 19661)

    let noneBody = try decodeJSONBody(try #require(requests[1].body))
    #expect(noneBody["thinking"]?["type"]?.stringValue == "disabled")
    #expect(noneBody["thinking"]?["token_budget"] == nil)

    let providerOptionsBody = try decodeJSONBody(try #require(requests[2].body))
    #expect(providerOptionsBody["thinking"]?["type"]?.stringValue == "enabled")
    #expect(providerOptionsBody["thinking"]?["token_budget"] == nil)

    let providerDefaultBody = try decodeJSONBody(try #require(requests[3].body))
    #expect(providerDefaultBody["thinking"] == nil)

    let unsupportedBody = try decodeJSONBody(try #require(requests[4].body))
    #expect(unsupportedBody["thinking"] == nil)
    #expect(unsupported.warnings == [
        AIWarning(
            type: "unsupported",
            feature: "reasoning",
            message: #"reasoning "unsupported-level" is not supported by this model."#
        )
    ])
}

@Test func coherePassesProviderAndRequestHeadersLikeUpstream() async throws {
    let chatTransport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}
    """))
    let chatProvider = try AIProviders.cohere(configuration: CohereProviderSettings(
        apiKey: "cohere-key",
        headers: ["Custom-Provider-Header": "provider-header-value"],
        transport: chatTransport
    ))
    _ = try await chatProvider.languageModel("command-r-plus").generate(LanguageModelRequest(
        messages: [.user("Hi")],
        headers: ["Custom-Request-Header": "request-header-value"]
    ))
    let chatRequest = try #require(await chatTransport.requests().first)
    #expect(chatRequest.headers["authorization"] == "Bearer cohere-key")
    #expect(chatRequest.headers["custom-provider-header"] == "provider-header-value")
    #expect(chatRequest.headers["Custom-Request-Header"] == "request-header-value")

    let streamTransport = RecordingTransport(response: sseResponse("""
    data: {"type":"message-start","id":"msg-1"}

    data: {"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}}

    """))
    let streamProvider = try AIProviders.cohere(configuration: CohereProviderSettings(
        apiKey: "cohere-key",
        headers: ["Custom-Provider-Header": "provider-header-value"],
        transport: streamTransport
    ))
    for try await _ in try streamProvider.languageModel("command-r-plus").stream(LanguageModelRequest(
        messages: [.user("Hi")],
        headers: ["Custom-Request-Header": "request-header-value"]
    )) {}
    let streamRequest = try #require(await streamTransport.requests().first)
    #expect(streamRequest.headers["custom-provider-header"] == "provider-header-value")
    #expect(streamRequest.headers["Custom-Request-Header"] == "request-header-value")

    let embeddingTransport = RecordingTransport(response: jsonResponse("""
    {"embeddings":{"float":[[0.1,0.2]]},"meta":{"billed_units":{"input_tokens":1}}}
    """))
    let embeddingProvider = try AIProviders.cohere(configuration: CohereProviderSettings(
        apiKey: "cohere-key",
        headers: ["Custom-Provider-Header": "provider-header-value"],
        transport: embeddingTransport
    ))
    _ = try await embeddingProvider.embeddingModel("embed-english-v3.0").embed(EmbeddingRequest(
        values: ["hello"],
        headers: ["Custom-Request-Header": "request-header-value"]
    ))
    let embeddingRequest = try #require(await embeddingTransport.requests().first)
    #expect(embeddingRequest.headers["custom-provider-header"] == "provider-header-value")
    #expect(embeddingRequest.headers["Custom-Request-Header"] == "request-header-value")
}

@Test func cohereUnknownFinishReasonMapsToOther() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"SOMETHING_NEW","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.finishReason == "other")
}
@Test func cohereLanguageMapsStandardOptionsResponseFormatAndToolChoice() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"thinking","thinking":"plan"},{"type":"text","text":"{\\"answer\\":\\"ok\\"}"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":6}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Return JSON.")],
        temperature: 0.2,
        topP: 0.7,
        topK: 9,
        presencePenalty: 0.1,
        frequencyPenalty: 0.2,
        seed: 42,
        maxOutputTokens: 32,
        stopSequences: ["END"],
        responseFormat: .json(
            schema: [
                "type": "object",
                "properties": ["answer": ["type": "string"]],
                "required": ["answer"]
            ],
            name: "ignored-by-cohere",
            description: "Ignored by Cohere"
        ),
        reasoning: "128",
        tools: [
            "lookup": [
                "type": "object",
                "description": "Look up a value.",
                "properties": ["value": ["type": "string"]],
                "required": ["value"]
            ],
            "unused": ["type": "object"]
        ],
        toolChoice: ["type": "tool", "toolName": "lookup"],
        providerOptions: [
            "cohere": [
                "thinking": [
                    "type": "enabled",
                    "tokenBudget": 64
                ],
                "safetyMode": "STRICT"
            ]
        ],
        extraBody: [
            "cohere": [
                "responseFormat": [
                    "type": "json_object",
                    "json_schema": ["type": "string"]
                ],
                "thinking": [
                    "type": "enabled",
                    "tokenBudget": 64
                ]
            ]
        ]
    ))

    #expect(result.text == #"{"answer":"ok"}"#)
    #expect(result.reasoning == "plan")
    #expect(result.usage?.totalTokens == 11)

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["temperature"]?.doubleValue == 0.2)
    #expect(body["p"]?.doubleValue == 0.7)
    #expect(body["k"]?.intValue == 9)
    #expect(body["presence_penalty"]?.doubleValue == 0.1)
    #expect(body["frequency_penalty"]?.doubleValue == 0.2)
    #expect(body["seed"]?.intValue == 42)
    #expect(body["max_tokens"]?.intValue == 32)
    #expect(body["stop_sequences"]?[0]?.stringValue == "END")
    #expect(body["response_format"]?["type"]?.stringValue == "json_object")
    #expect(body["response_format"]?["json_schema"]?["type"]?.stringValue == "object")
    #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["thinking"]?["token_budget"]?.intValue == 64)
    #expect(body["tools"]?.arrayValue?.count == 1)
    #expect(body["tools"]?[0]?["type"]?.stringValue == "function")
    #expect(body["tools"]?[0]?["function"]?["name"]?.stringValue == "lookup")
    #expect(body["tools"]?[0]?["function"]?["description"]?.stringValue == "Look up a value.")
    #expect(body["tools"]?[0]?["function"]?["parameters"]?["properties"]?["value"]?["type"]?.stringValue == "string")
    #expect(body["tool_choice"]?.stringValue == "REQUIRED")
    #expect(body["safetyMode"] == nil)
    #expect(body["cohere"] == nil)
    #expect(body["responseFormat"] == nil)
    #expect(body["toolChoice"] == nil)
}
@Test func cohereLanguageExtractsUserFilesIntoDocuments() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":3,"output_tokens":1}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-r-plus")

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .text("What do these documents say?"),
            .file(mimeType: "text/plain", data: Data("First document content".utf8), filename: "doc1.txt"),
            .file(mimeType: "application/json", data: Data("{\"key\":\"value\"}".utf8), filename: "data.json")
        ])
    ]))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["messages"]?[0]?["role"]?.stringValue == "user")
    #expect(body["messages"]?[0]?["content"]?.stringValue == "What do these documents say?")
    #expect(body["documents"]?.arrayValue?.count == 2)
    #expect(body["documents"]?[0]?["data"]?["text"]?.stringValue == "First document content")
    #expect(body["documents"]?[0]?["data"]?["title"]?.stringValue == "doc1.txt")
    #expect(body["documents"]?[1]?["data"]?["text"]?.stringValue == "{\"key\":\"value\"}")
    #expect(body["documents"]?[1]?["data"]?["title"]?.stringValue == "data.json")
}
@Test func cohereRejectsUnsupportedDocumentMediaTypeLikeUpstream() async throws {
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: RecordingTransport(responses: [])))
    let model = try provider.languageModel("command-r-plus")

    await #expect(throws: AIError.invalidArgument(
        argument: "files",
        message: "Media type 'application/pdf' is not supported. Supported media types are: text/* and application/json."
    )) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [
                .file(mimeType: "application/pdf", data: Data("PDF-like content".utf8), filename: "doc.pdf")
            ])
        ]))
    }
}
@Test func cohereLanguageKeepsImagesInlineWhileExtractingDocuments() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":3,"output_tokens":1}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-r-plus")

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .text("Use both."),
            .file(mimeType: "image/png", data: Data([0, 1, 2, 3]), filename: "image.png"),
            .data(mimeType: "image/*", data: Data([0, 1, 2])),
            .file(mimeType: "text/plain", data: Data("Document text".utf8), filename: "note.txt")
        ])
    ]))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["type"]?.stringValue == "text")
    #expect(content[0]["text"]?.stringValue == "Use both.")
    #expect(content[1]["type"]?.stringValue == "image_url")
    #expect(content[1]["image_url"]?["url"]?.stringValue == "data:image/png;base64,AAECAw==")
    #expect(content[2]["type"]?.stringValue == "image_url")
    #expect(content[2]["image_url"]?["url"]?.stringValue == "data:image/jpeg;base64,AAEC")
    #expect(body["documents"]?.arrayValue?.count == 1)
    #expect(body["documents"]?[0]?["data"]?["text"]?.stringValue == "Document text")
    #expect(body["documents"]?[0]?["data"]?["title"]?.stringValue == "note.txt")
}
@Test func cohereProviderDefinedToolsAreSkippedWithWarning() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":1,"output_tokens":1}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        tools: [
            "lookup": [
                "type": "object",
                "properties": ["value": ["type": "string"]]
            ],
            "cohere.search": [
                "type": "provider",
                "id": "cohere.search"
            ]
        ]
    ))

    #expect(result.warnings == [
        AIWarning(type: "unsupported", feature: "provider-defined tool cohere.search")
    ])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let tools = try #require(body["tools"]?.arrayValue)
    #expect(tools.count == 1)
    #expect(tools[0]["function"]?["name"]?.stringValue == "lookup")
}
@Test func cohereLanguageParsesToolCallsAndNullArguments() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[],"tool_calls":[{"id":"currentTime_tf4dywn8wgnk","type":"function","function":{"name":"currentTime","arguments":"null"}}]},"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":3,"output_tokens":2}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("What time is it?")]))

    #expect(result.text == "")
    #expect(result.finishReason == "tool-calls")
    #expect(result.usage?.totalTokens == 5)
    #expect(result.toolCalls.count == 1)
    #expect(result.toolCalls[0].id == "currentTime_tf4dywn8wgnk")
    #expect(result.toolCalls[0].name == "currentTime")
    #expect(result.toolCalls[0].arguments == "{}")
}
@Test func cohereLanguageSerializesToolCallsAndToolResultMessages() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-2","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":4,"output_tokens":1}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-a-03-2025")

    _ = try await model.generate(LanguageModelRequest(messages: [
        .assistant(toolCalls: [
            AIToolCall(id: "call_weather", name: "weather", arguments: #"{"city":"Tokyo"}"#)
        ]),
        .toolResult(AIToolResult(
            toolCallID: "call_weather",
            toolName: "weather",
            result: ["forecast": "sunny"]
        )),
        .toolResult(AIToolResult(
            toolCallID: "call_denied",
            toolName: "deny",
            result: ["raw": "fallback"],
            modelOutput: ["type": "execution-denied"]
        ))
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["messages"]?[0]?["role"]?.stringValue == "assistant")
    #expect(body["messages"]?[0]?["content"] == nil)
    #expect(body["messages"]?[0]?["tool_calls"]?[0]?["id"]?.stringValue == "call_weather")
    #expect(body["messages"]?[0]?["tool_calls"]?[0]?["function"]?["name"]?.stringValue == "weather")
    #expect(body["messages"]?[0]?["tool_calls"]?[0]?["function"]?["arguments"]?.stringValue == #"{"city":"Tokyo"}"#)
    #expect(body["messages"]?[1]?["role"]?.stringValue == "tool")
    #expect(body["messages"]?[1]?["tool_call_id"]?.stringValue == "call_weather")
    #expect(body["messages"]?[1]?["content"]?.stringValue == #"{"forecast":"sunny"}"#)
    #expect(body["messages"]?[2]?["role"]?.stringValue == "tool")
    #expect(body["messages"]?[2]?["tool_call_id"]?.stringValue == "call_denied")
    #expect(body["messages"]?[2]?["content"]?.stringValue == "Tool execution denied.")
}
@Test func cohereLanguageMapsCitationSources() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"generation_id":"gen-1","message":{"role":"assistant","content":[{"type":"text","text":"AI helps automate work."}],"citations":[{"start":9,"end":17,"text":"automate","sources":[{"type":"document","id":"doc:0","document":{"id":"doc:0","text":"AI helps automate work.","title":"benefits.txt"}}],"type":"TEXT_CONTENT"}]},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":3,"output_tokens":4}}}
    """))
    let provider = try AIProviders.cohere(configuration: CohereProviderSettings(apiKey: "cohere-key", transport: transport))
    let model = try provider.languageModel("command-r-plus")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("What are AI benefits?")]))

    #expect(result.text == "AI helps automate work.")
    #expect(result.sources.count == 1)
    #expect(result.sources[0].id == "cohere-citation-0")
    #expect(result.sources[0].sourceType == "document")
    #expect(result.sources[0].title == "benefits.txt")
    #expect(result.sources[0].mediaType == "text/plain")
    #expect(result.sources[0].providerMetadata["cohere"]?["start"]?.intValue == 9)
    #expect(result.sources[0].providerMetadata["cohere"]?["end"]?.intValue == 17)
    #expect(result.sources[0].providerMetadata["cohere"]?["text"]?.stringValue == "automate")
    #expect(result.sources[0].providerMetadata["cohere"]?["citationType"]?.stringValue == "TEXT_CONTENT")
    #expect(result.sources[0].providerMetadata["cohere"]?["sources"]?[0]?["document"]?["title"]?.stringValue == "benefits.txt")
    #expect(result.sources[0].rawValue?["sources"]?[0]?["id"]?.stringValue == "doc:0")
}