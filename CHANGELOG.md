# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-10

### Breaking Changes
- Derived transforms no longer carry coordinate systems. `copy()`, `as_affine()`, `as_st()` and the optimized `__mul__`/`__rmul__` products are system-less by default, where they previously inherited `from_cs`/`to_cs` and could silently overwrite canonical graph edges or add phantom shortcuts
- Systems on a copy are opt-in: `copy(from_cs=..., to_cs=...)` or `copy(system_names=<dict or suffix>)`, resolved all-or-nothing with no auto-generated names
- A registered copy must introduce at least one new coordinate system, so it can never replace or shadow an existing canonical edge; names that collide with the graph, or two existing systems given as both `from_cs` and `to_cs`, are rejected
- `Image.__getitem__` applies its slices to spatial axes rather than the leading image axes, matching its documented behaviour. Passing more slices than there are spatial dimensions is now an error instead of being silently ignored
- `CallbackRegistry` compares callbacks by identity rather than equality. This changes nothing for bound methods, which CPython already compares by identity, but differs for non-method callables defining `__eq__`

### New features
- `Transform.map()` accepts Qt geometry types directly — `QPointF`, `QVector3D`, and lists or tuples of them — via exact-type dispatch
- `import_qt_core()` exposed from `coorx.qt` for consistent Qt access
- `as_affine(from_cs=..., to_cs=...)` assigns systems to the result, backed by a new `_as_affine()` subclass hook
- `CompositeTransform.copy()` resolves and registers a whole chain, so an entire path can be frozen as new edges

### Changes
- A `None` coordinate system acts as a wildcard in multiplication and composite chain validation, so system-less derived transforms still compose with canonical ones; the check it skips is deferred to map time
- `CompositeTransform.copy()` flattens nested chains before resolving systems, so a chain containing a `CompositeTransform` can name its interior systems instead of raising

### Fixes
- `CompositeTransform.copy(system_names=...)` registers each canonical edge once. A chain visiting the same pair of systems twice — a transform together with its inverse — previously registered that edge twice, overwriting the frozen chain's own first edge
- `make_crop_transform` pairs each spatial slice with its own image axis when measuring extents

### Performance
- `CallbackRegistry` operations are O(1) per callback rather than O(n). With 4000 weak subscribers on one transform, registering fell from 1.57s to 0.005s and collecting from 2.01s to 0.002s

### Internal
- Notebook tests run in a kernel isolated from the developer's IPython profile, so personal startup scripts cannot inject extra cell outputs
- matplotlib is pinned in CI so the notebook's pixel-exact image comparisons are reproducible (see #61)

## [2.0.0] - 2026-01

### Breaking Changes
- Transform `__getstate__()` serialization format changed - saved transforms from v1.x will not be compatible
- Transform `save_state()` serialization format changed - saved transforms from v1.x will not be compatible
- `scale()` method renamed to `zoom()` across STTransform, AffineTransform, and SRT3DTransform
- SRT3DTransform getters converted to properties: use `.scale`, `.rotation`, `.offset` instead of getter methods
- Setter methods replaced with property setters for cleaner API
- Transform initialization system completely refactored with new parameter specification system
- `create_transform()` function signature changed: no longer accepts `params` argument
- Transform internal parameter handling migrated to new declarative specification system
- CompositeTransform no longer accepts star-args for initialization; use a list of transforms instead

### New features
- New event system (`events.py`) with automatic callback lifecycle management
- Parameter specification system (`params.py`) for declarative transform parameter definitions
- QMatrix4x4 conversion methods (`to_qmatrix4x4()` and `from_qmatrix4x4()`) for Qt framework integration
- `AffineTransform.from_matrix()` class method for convenient matrix-based construction
- AffineTransform now supports disparate dimensionality between input and output
- `keep_reference` parameter in `add_change_callback()` to control weak vs strong reference behavior
- `duplicates` parameter in `add_change_callback()` to control duplicate callback handling

### Changes
- Callback system now uses weak references by default to prevent memory leaks from bound methods
- All transform classes migrated to new parameter specification system
- `Image.copy()` optimized to only copy pixel data when actually modified
- Transform API standardized across all classes with consistent `zoom()` method and property access
- `zoom()` methods now have optional `center` parameter (defaults to origin)
- All tests updated to use new property-based API
- Callback invocation protected by mutex to prevent race conditions
- Dead callback references automatically cleaned up during `_update()`
- PyQtGraph reload compatibility improved for safer module reloading

### Fixes
- Memory leaks in change callback system when using bound methods
- Duplicate callback invocations prevented through internal tracking
- Image copy now creates truly independent copies of image data
- Rotation axis double-wrapping bug in transform composition
- Homogeneous coordinate handling in framework conversions

### Internal
- Major refactoring of transform base class parameter handling
- Event callback system extracted into dedicated module
- Test suite expanded by ~1,100 lines
- Code formatting standardized to `black -S -l 100` in a few modules

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