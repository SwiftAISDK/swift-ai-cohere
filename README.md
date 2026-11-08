<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-cohere 1.2.0

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-cohere.git", from: "1.2.0")
```

Products: `SwiftAICohere`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# SwiftAICohere

Swift-native port of `@ai-sdk/cohere`. Maintained in the canonical SwiftAISDK repository and exported as an independent Swift package.

The canonical repository owns all source edits. This package owns its provider
configuration and wire behavior, using only the declared shared dependencies.
Existing upstream baselines, capabilities and documented gaps remain unchanged.
See `Porting/map.json` and the canonical provider guide.

See [provider usage and capabilities](https://github.com/SwiftAISDK/SwiftAISDK/blob/main/docs-site/scripts/provider-guides/cohere.md) for authentication, options and supported surfaces.
## Behavior notes

Partial document citations are accepted, including streaming `citation-start` events. Tool-only citations remain in the raw response and do not produce document sources. Response metadata prefers the v2 `id`; missing token usage stays unknown, while billed units remain in raw usage. Tool replay preserves valid JSON argument text and quotes malformed text as a JSON string.
