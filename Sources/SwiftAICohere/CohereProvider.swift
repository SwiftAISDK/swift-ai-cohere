import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public struct CohereProviderSettings: Sendable {
    public var apiKey: String?
    public var baseURL: String?
    public var headers: [String: String]
    public var queryParams: [String: String]
    public var transport: any AITransport

    public init(apiKey: String? = nil, baseURL: String? = nil, headers: [String: String] = [:],
                queryParams: [String: String] = [:], transport: any AITransport = URLSessionTransport.shared) {
        self.apiKey = apiKey; self.baseURL = baseURL; self.headers = headers
        self.queryParams = queryParams; self.transport = transport
    }
}

public final class CohereProvider: AIProvider, Sendable {
    public let providerID = "cohere"
    public let supportedCapabilities: Set<ModelCapability> = [.language, .embedding, .reranking]
    private let http: AIProviderHTTPClient

    public init(configuration: CohereProviderSettings = .init()) throws {
        let names = ["COHERE_API_KEY"]
        guard let key = configuration.apiKey ?? environmentValue(names) else {
            throw AIError.missingAPIKey(provider: "cohere", environmentVariables: names)
        }
        var headers = configuration.headers
        headers["Authorization"] = headers["Authorization"] ?? "Bearer \(key)"
        headers = withUserAgentSuffix(headers, "ai-sdk-cohere/4.0.69")
        let baseURL = withoutTrailingSlash(configuration.baseURL ?? "https://api.cohere.com/v2")
        http = AIProviderHTTPClient(providerID: "cohere", headers: headers, transport: configuration.transport,
            url: { _, path in try providerURL("\(baseURL)\(path)", queryParams: configuration.queryParams) })
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        CohereLanguageModel(modelID: modelID, http: http)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        CohereEmbeddingModel(modelID: modelID, http: http)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .image, modelID: modelID)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .speech, modelID: modelID)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .video, modelID: modelID)
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        CohereRerankingModel(modelID: modelID, http: http)
    }

    public func chatModel(_ modelID: String) throws -> any LanguageModel { try languageModel(modelID) }
    public func reranking(_ modelID: String) throws -> any RerankingModel { try rerankingModel(modelID) }
}

public extension AIProviders {
    static func cohere(configuration: CohereProviderSettings) throws -> CohereProvider {
        try CohereProvider(configuration: configuration)
    }
}

struct CohereHTTPConfiguration: AIProviderHTTPConfiguration {
    var http: AIProviderHTTPClient
}
