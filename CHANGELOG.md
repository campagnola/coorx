# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-09-23

### Added
- N-dimensional image support with spatial axis specification
- Bilinear transform for 2D coordinate mapping
- 2D homography transform implementation
- Comprehensive nonlinear transform edge case testing
- Enhanced coordinate system graph search functionality
- Automatic mapping between coordinate systems via graph pathfinding

### Changed
- LogTransform specification updated: None base = identity, negative base = inverse
- Image class spatial attribute naming for clarity (`spatial_shape`, `spatial_axes`)
- Simplified CompositeTransform behavior and implementation
- Enhanced VectorArray initialization to only accept two points for consistency
- Improved coordinate system functionality and image rotation handling
- Floating-point precision improvements in polar and log transforms

### Fixed
- LogTransform edge cases and test specifications
- Nonlinear transform edge case behavior and testing
- Image coordinate system parameter handling and typos
- Spurious antialiasing errors in image tests through better tolerance
- VectorArray initialization validation and error messages
- Image test notebook execution and pixel comparison accuracy

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