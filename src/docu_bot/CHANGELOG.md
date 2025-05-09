# Chanelog `Docu-Bot`

Changelog for `Docu-Bot` project, documenting all released changes made to the project.

## [2.2.6] - 2025-04-08
### Added
- Dockerfile for easy deployment

## [2.2.6] - 2025-03-15
### Fixed
- Linux compatibility (Issue #10)

## [2.2.5] - 2025-03-11
### Fixed
- Script Path

## [2.2.4] - 2025-03-11
### Changed
- Refactored the code for better maintainability
### Added
- Component tests

## [2.2.3] - 2025-02-28
### Added
- Keyphrase Retrieval
- Added More results
### Fixed
- Added fallbacks when extraction fails
- Unaccounted packages
- Added fallback for git

## [2.2.2] - 2025-02-22
### Added
- NER and Theme Retrieval
- Added result visualization
- Timing for generating answers
### Fixed
- Fixed Typing hints

## [2.2.1] - 2025-02-21
### Changed
- Some optimization of prompts
- Added support for more embeddings
### Fixed
- Fixed paths for git repositories with dots in the name

## [2.2.0] - 2025-02-16
### Changed
- Separated components for better maintainability
- Added support for more models
- Added notebooks for testing and evaluation

## [2.1.0] - 2024-12-07
### Added
- Chat interface for better interaction with the bot
- Judging the quality of the answer
- Reranking found documents based on the quality of the answer

## [2.0.0] - 2024-11-23
### Changed
- OpenAI API only solution, custom models to be used with  `https://github.com/jinymusim/serve-model` project

## [1.2.1] - 2024-04-26
### Fixed
- Fixed branch selection

## [1.2.0] - 2024-04-26
### Added
- Quick branch submition without need to specify redirect

## [1.1.2] - 2024-04-25
### Changed
- Automatic selection of inserted branch

## [1.1.1] - 2024-04-25
### Fixed
- Fixed repository selection

## [1.1.0] - 2024-04-24
### Added
- Added `Redirects` support

## [1.0.0] - 2024-04-21
### Added
- Initial release of Docu-Bot