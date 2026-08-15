# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.4](https://github.com/inmzhang/ticit/compare/v0.2.3...v0.2.4) - 2026-08-15

### Added

- *(bench)* add PPVM MSD Python comparison

### Other

- *(deps)* update astral-sh/setup-uv action to v10.0.1 ([#19](https://github.com/inmzhang/ticit/pull/19))

## [0.2.3](https://github.com/inmzhang/ticit/compare/v0.2.2...v0.2.3) - 2026-08-13

### Added

- *(gpu)* pin measurement parities
- *(sampler)* pin measurement parities

### Fixed

- *(gpu)* default to retained records
- *(ccz)* consolidate fixture and benchmark fixes

### Other

- *(deps)* apply dashboard updates

## [0.2.2](https://github.com/inmzhang/ticit/compare/v0.2.1...v0.2.2) - 2026-08-11

### Other

- *(planner)* reduce large-frame peak memory

## [0.2.1](https://github.com/inmzhang/ticit/compare/v0.2.0...v0.2.1) - 2026-08-11

### Added

- *(sampler)* add reference normalization
- *(gpu)* export exact-k shot records ([#15](https://github.com/inmzhang/ticit/pull/15))

### Other

- *(lint)* fix typos false positives
- *(format)* rename circuit extension to .tic
- *(planner)* build tableaus from final rows
- *(bench)* publish normalized CPU results
- *(sampler)* skip empty reference shots
- *(api)* use postselection masks
- *(gpu)* drop the never-taken detector tail loop
- remove scaffolding and colocate tests

### Added

- CPU-computed reference samples and syndrome normalization for CPU and GPU sampling
- Reference-normalized benchmark defaults

## [0.2.0](https://github.com/inmzhang/ticit/compare/v0.1.3...v0.2.0) - 2026-08-10

### Added

- *(simulator)* [**breaking**] replay parsed circuits

### Other

- fix API link consistency
- compare tableau and sampler performance

## [0.1.3](https://github.com/inmzhang/ticit/compare/v0.1.2...v0.1.3) - 2026-08-09

### Fixed

- *(simulator)* gate popcnt probe to x86

### Other

- *(simulator)* specialize frame and replay

## [0.1.2](https://github.com/inmzhang/ticit/compare/v0.1.1...v0.1.2) - 2026-08-09

### Other

- update package descriptions
- fix manual PyPI publishing
- register PyPI workflow
- Merge pull request #11 from inmzhang/release-plz-2026-08-09T13-37-48Z

## [0.1.1](https://github.com/inmzhang/ticit/compare/v0.1.0...v0.1.1) - 2026-08-09

### Fixed

- allow x86-only fields on ARM

### Other

- *(deps)* update actions/checkout action to v7
- improve tableau simulator docs
- avoid duplicate PR runs
- cache Rust dependencies
- simplify installation headings
- skip GPU features on hosted runners
