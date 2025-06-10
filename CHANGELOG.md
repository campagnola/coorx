# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-09

### Added
- Vector and VectorArray classes for coordinate math operations
- Point and PointArray subtraction operations returning Vector/VectorArray
- Vector initialization with ndarray support
- `mapped_to()` method for Vector and VectorArray classes enabling automatic coordinate system mapping
- Image transformation support with test notebook
- ASCII image diff visualization for better test output comparison
- Comprehensive development documentation in CLAUDE.md
- Enhanced Jupyter notebook testing infrastructure with pristine state checking
- Support for visual diff comparisons in notebook tests

### Changed
- Point subtraction now returns Vector instead of Point (breaking change)
- Improved coordinate system consistency in API
- Enhanced test coverage with parameterized transform combination tests
- Updated testing dependencies and GitHub Actions configuration

### Fixed
- Image coordinate system handling in transformations
- Notebook test execution and output comparison reliability
- Spurious antialiasing errors in image tests through tolerance adjustments

## [1.0.5] - Previous Release

### Features
- Object-oriented coordinate system transforms
- Linear transforms: Null, Translation, Scale+Translate, Affine, SRT3D, Transpose
- Nonlinear transforms: Log, Polar
- Composite transform chaining with automatic simplification
- Coordinate system graph for automatic pathfinding between coordinate systems
- Framework compatibility with ITK, Qt, scikit-image, and VisPy
- Bidirectional mapping support (forward and inverse)
- Point and PointArray coordinate classes with system awareness