# Changelog

## 1.1.0

- Native UTF-8 Data document conversion already covers upstream byte/base64 decoding; new fixtures prove Unicode and literal base64-looking text preserve the intended text/title. Portable thinking remains enabled when no budget can be mapped; it retains the warning and omits an invented budget. Reviewed upstream: @ai-sdk/cohere 4.0.69 to 4.0.73. Validation passed: node Scripts/test-packages.mjs (all 52 owning package suites and the compatibility facade); node Scripts/sdk.mjs generate --check; node --test Scripts/tests/*.test.cjs (27 passed); docs-site check and build (100 pages). Live provider calls were not run.
  Compatibility: Minor release for additive native API; existing static settings and call sites remain compatible. Updated direct dependency minimums are recorded in the manifest. Existing documented JS/framework and native product gaps remain; version-delta review does not imply full parity or live-service testing. Explicit custom authentication skips an unused credential supplier, a deliberate native callback-order adaptation.
