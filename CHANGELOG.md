# Changelog

## 0.1.0 - 2026-05-21

Initial public release.

- Add fast GraphQL-backed run metadata discovery.
- Cache run metadata, sampled history metrics, and table artifact rows as Parquet.
- Add selective `config_keys` support so large configs are not repeated into every DataFrame row by default.
- Add public CleanRL and W&B Tables examples.
- Add offline tests and GitHub Actions CI.
