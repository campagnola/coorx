# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands likely require running python from a particular environment. If this is not specified in an agent instruction file, then prompt the user.

### Testing
- **Run all tests**: `python -m pytest coorx/tests`
- **Run specific test file**: `python -m pytest coorx/tests/test_transforms.py`
- **Run single test**: `python -m pytest coorx/tests/test_transforms.py::CompositeTransform::test_as_affine`
- **Collect tests only**: `python -m pytest --collect-only`

## Architecture

Coorx implements object-oriented coordinate system transforms with an emphasis on composability and automatic mapping between coordinate systems.

### Core Transform Hierarchy
- **Transform** (`base_transform.py`): Base class defining `map()` and `imap()` methods, coordinate system management, and transform composition via `__mul__`
- **Linear transforms** (`linear.py`): NullTransform, TTransform (translation), STTransform (scale+translate), AffineTransform, SRT3DTransform, TransposeTransform
- **Nonlinear transforms** (`nonlinear.py`): LogTransform, PolarTransform  
- **CompositeTransform** (`composite.py`): Chains multiple transforms together, supports automatic simplification

### Coordinate System Graph (`systems.py`)
- **CoordinateSystemGraph**: Manages relationships between named coordinate systems
- **CoordinateSystem**: Represents individual coordinate systems with dimensionality
- Enables automatic pathfinding between coordinate systems via transform chains
- Global graph registry with "default" graph for most operations

### Transform Features
- **Transform flags**: Linear, Orthogonal, NonScaling, Isometric properties guide optimization
- **Bidirectional mapping**: All transforms support forward (`map`) and inverse (`imap`) operations
- **Matrix representation**: Linear transforms provide `full_matrix` and `as_affine()` methods
- **Framework compatibility**: Conversion methods for ITK, Qt, scikit-image, and VisPy

### Coordinate Arrays (`coordinates.py`)
- **Point/PointArray**: Position coordinates that know their coordinate system
- **Vector/VectorArray**: Direction vectors with coordinate system awareness
- **Automatic mapping**: `mapped_to()` method uses coordinate system graph for automatic transform lookup

### Image Support (`image.py`)
- Image class that tracks coordinate system of pixel data
- Integrates with transform system for spatial transformations
- Basic manipulations like crop, scale, and rotate track transformations to allow mapping between pixels of original and final images

### Testing Infrastructure
- Custom pytest plugin (`conftest.py`) for Jupyter notebook testing
- Pristine state checking and execution comparison for `.ipynb` files
- Visual diff support for image outputs using ASCII art representation
- Many parameterized tests covering all transform combinations
