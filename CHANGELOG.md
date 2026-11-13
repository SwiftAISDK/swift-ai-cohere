# Changelog

## 1.2.1

- Refresh the versioned request identity while retaining the current native request, response and stream behavior. Internal upstream model-catalog reorganizations preserve the existing native policies. Reviewed npm delta: @ai-sdk/cohere 4.0.78 to 4.0.83. Validation: `node Scripts/test-packages.mjs` passed all 52 owning suites and 1731 facade tests; tooling 27/27, generation/isolation checks, standalone exports and 106 minimum/latest consumers, documentation check and 101-page build passed. Publication requires green exact-source Swift 6.3.3 CI. Paid/live verification and full-package parity are not claimed.
  Compatibility: Compatible behavior/header correction. Original public selectors are preserved. npm baselines and Swift SemVer are independent; this reviewed delta does not establish full-package or live-service parity.
