# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.9.2] - UNRELEASED

### Added

* Make parameters optionally serializable/deserializable (if `serde` feature is enabled).

### Changed

* Switch to `orp` version `0.9.2`.


## [0.9.1] - 2025-03-09

### Added

* Embeddings extraction is now more flexible in order to support different models. For now, the parameters allow to configure the name of the tensor output, and to select one of two modes: `Raw` or `Token` (see crate documentation).
* Added two examples for embeddings using `gte-multilingual-base`, that leverage these new options.

### Changed

* Move the examples into a more standard location (`./examples` instead of `./src/examples`).


## [0.9.0] - 2025-02-01

Initial release.
