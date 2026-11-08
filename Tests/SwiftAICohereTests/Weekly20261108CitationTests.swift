import Foundation
import Testing
@testable import SwiftAICohere
import SwiftAIProvider

@Suite("Weekly20261108 Cohere partial citations and usage")
struct Weekly20261108CitationTests {
    @Test(arguments: [
        JSONValue.object(["id": "v2", "generation_id": "v1"]),
        ["generation_id": "v1"], [:]
    ])
    func partialCitationResponsePrefersV2IDAndAcceptsMissingUsage(_ fields: JSONValue) async throws {
        var raw = try #require(fields.objectValue)
        raw["message"] = ["role": "assistant", "content": [["type": "text", "text": "ok"]], "citations": [
            ["sources": [["type": "tool", "tool_output": [:]]]],
            ["sources": [["tool_output": [:]], ["document": ["title": "", "custom": 42]]]]
        ]]
        raw["finish_reason"] = "COMPLETE"
        let transport = RecordingTransport(response: .init(statusCode: 200, body: try JSONEncoder().encode(JSONValue.object(raw))))
        let provider = try CohereProvider(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.languageModel("command").generate(.init(messages: [.user("Hello")]))
        #expect(result.responseMetadata.id == (fields["id"]?.stringValue ?? fields["generation_id"]?.stringValue))
        #expect(result.sources.count == 1)
        #expect(result.sources.first?.title == "Document")
        #expect(result.sources.first?.providerMetadata["cohere"]?["sources"]?[1]?["document"]?["custom"] == 42)
        #expect(result.usage?.inputTokens == nil)
        #expect(result.usage?.outputTokens == nil)
    }

    @Test(arguments: [#"{"city":"Tokyo"}"#, "null", "42", "raw arguments\n\"quoted\"", ""])
    func toolArgumentsPreserveJSONAndQuoteMalformedText(_ input: String) throws {
        let part = AIContentPart.toolCall(.init(id: "call", name: "lookup", arguments: input))
        let wire = try #require(cohereAssistantToolCallJSON(part)?["function"]?["arguments"]?.stringValue)
        if let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(input.utf8)) {
            #expect(wire == input)
            #expect(try JSONDecoder().decode(JSONValue.self, from: Data(wire.utf8)) == parsed)
        } else {
            #expect(try JSONDecoder().decode(JSONValue.self, from: Data(wire.utf8)) == .string(input))
        }
    }
    @Test func toolOnlyCitationsAreOmittedAndFirstDocumentIsSelected() {
        let citations: JSONValue = [["sources": [["type": "tool", "tool_output": ["answer": 1]]]], ["start": .null, "sources": [["type": "tool", "id": "tool"], ["document": ["title": "Document title", "custom": 42]]]]]
        let sources = cohereSources(from: citations)
        #expect(sources.count == 1)
        #expect(sources.first?.title == "Document title")
        #expect(sources.first?.providerMetadata["cohere"]?["start"] == nil)
        #expect(sources.first?.providerMetadata["cohere"]?["sources"]?[1]?["document"]?["custom"] == 42)
    }
    @Test func billedUnitsNeverStandInForTokensAndRawUsageSurvives() {
        let usage: JSONValue = ["billed_units": ["input_tokens": 100], "cached_tokens": 4]
        let result = cohereTokenUsage(from: ["usage": usage])
        #expect(result?.inputTokens == nil)
        #expect(result?.inputTokensNoCache == nil)
        #expect(result?.inputTokensCacheRead == 4)
        #expect(result?.rawValue == usage)
    }
    @Test func streamEmitsSingleAndArrayCitationShapes() async throws {
        let transport = RecordingTransport(response: sseResponse("""
        data: {"type":"citation-start","delta":{"message":{"citations":{"sources":[{"document":{"title":"A"}}]}}}}

        data: {"type":"citation-start","delta":{"message":{"citations":[{"sources":[{"tool_output":{}}]},{"sources":[{"document":{"title":"B"}}]}]}}}

        data: {"type":"message-end","delta":{"finish_reason":"COMPLETE"}}

        """))
        let provider = try CohereProvider(configuration: .init(apiKey: "key", transport: transport))
        var sources: [AISource] = []
        for try await part in try provider.languageModel("command").stream(.init(messages: [.user("Hello")])) {
            if case let .source(source) = part { sources.append(source) }
        }
        #expect(sources.map(\.title) == ["A", "B"])
        #expect(Set(sources.map(\.id)).count == 2)
    }
}
