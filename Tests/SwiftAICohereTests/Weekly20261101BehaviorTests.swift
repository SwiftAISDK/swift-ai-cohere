import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAICohere

@Suite struct Weekly20261101CohereBehaviorTests {
    @Test(arguments: ["This is file content", "Résumé: café, 日本語", "SGVsbG8="])
    func nativeBytesPreserveDecodedDocumentsAndLiteralBase64LookingText(_ text: String) async throws {
        let bytes = try #require(Data(base64Encoded: Data(text.utf8).base64EncodedString()))
        let transport = RecordingTransport(response: jsonResponse(#"{"message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE"}"#))
        let provider = try CohereProvider(configuration: .init(apiKey: "key", transport: transport))
        _ = try await provider.languageModel("command-a").generate(.init(messages: [.init(role: .user, content: [.text("Read"), .file(mimeType: "text/plain", data: bytes, filename: "test.txt")])]))
        let body = try decodeJSONBody(try #require(await transport.requests().first?.body))
        #expect(body["documents"]?[0]?["data"] == ["text": .string(text), "title": "test.txt"])
    }

    @Test func unsupportedBudgetRetainsEnabledThinking() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"finish_reason":"COMPLETE"}"#))
        let model = try CohereProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("command-a-reasoning-08-2025")
        let result = try await model.generate(.init(messages: [.user("Hi")], reasoning: "unsupported-level"))
        let body = try decodeJSONBody(try #require(await transport.requests().first?.body))
        #expect(body["thinking"] == ["type": "enabled"])
        #expect(result.warnings.contains { $0.feature == "reasoning" })
    }
}
