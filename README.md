<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-cohere 1.0.0

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-cohere.git", from: "1.0.0")
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
