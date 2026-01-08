import contextlib
import pickle

import numpy as np
import pytest
from coorx import CompositeTransform
from coorx import create_transform, Point
from coorx.coordinates import PointArray
from coorx.linear import STTransform
from coorx.systems import CoordinateSystemGraph
from pytest import raises

missing_tr = "No transform defined linking"
missing_cs = "No coordinate system named"
wrong_ndim = r"is \dD \(expected \dD\)"
wrong_system = "maps from system"
mult_impossible = "Cannot multiply transforms with different inner coordinate systems"
comp_impossible = "does not map to"


def test_coordinate_systems():
    default_graph = CoordinateSystemGraph.get_graph(None)
    # these coordinate system names are effectively global
    pt_cs1 = Point([0, 0], "2d-cs1")
    parr_cs2 = PointArray([[1, 1], [1, 0], [0, 1], [1, 1]], "2d-cs2")

    assert pt_cs1.system is default_graph.system("2d-cs1")
    assert np.all(pt_cs1.coordinates == [0, 0])
    with raises(TypeError):
        pt_cs1.mapped_to("2d-cs2")
    with raises(NameError, match=missing_cs):
        parr_cs2.mapped_to("nonexistent_cs")
    with raises(TypeError):
        Point([0, 0, 0], "2d-cs1")  # wrong ndim

    cs1_to_cs2 = STTransform(scale=[3, 2], offset=[10, 20], from_cs="2d-cs1", to_cs="2d-cs2")

    assert default_graph.transform("2d-cs1", "2d-cs2") is cs1_to_cs2
    assert default_graph.transform("2d-cs2", "2d-cs1") is cs1_to_cs2.inverse

    pt_cs2 = cs1_to_cs2.map(pt_cs1)

    with raises(TypeError, match=wrong_system):
        cs1_to_cs2.map(pt_cs2)
    with raises(TypeError, match=wrong_system):
        cs1_to_cs2.inverse.map(pt_cs1)

    assert pt_cs2.system is default_graph.system("2d-cs2")
    assert cs1_to_cs2.imap(pt_cs2).system is default_graph.system("2d-cs1")
    assert np.all(pt_cs2.coordinates == np.array([10, 20]))
    assert np.all(pt_cs1.coordinates == cs1_to_cs2.inverse.map(pt_cs2).coordinates)

    # composites and their inverses
    loop = CompositeTransform([cs1_to_cs2, cs1_to_cs2.inverse])
    assert loop.map(pt_cs1) == pt_cs1
    assert loop.inverse.map(pt_cs1) == pt_cs1

    # pickle point with CS
    assert pickle.loads(pickle.dumps(pt_cs1)) == pt_cs1

    # pickle transform with CS
    cs1_to_cs2_p = pickle.loads(pickle.dumps(cs1_to_cs2))
    assert cs1_to_cs2_p == cs1_to_cs2
    assert cs1_to_cs2_p.systems == cs1_to_cs2.systems


def test_mapped_to():
    CoordinateSystemGraph.get_graph(None)
    cs1_to_cs2 = STTransform(scale=[3, 2], offset=[10, 20], from_cs="2d-cs1", to_cs="2d-cs2")
    cs2_to_cs3 = STTransform(scale=[1, 1], offset=[0, 0], from_cs="2d-cs2", to_cs="2d-cs3")

    pt_cs1 = Point([0, 0], "2d-cs1")
    assert np.allclose(pt_cs1.mapped_to("2d-cs1"), pt_cs1)
    assert np.allclose(pt_cs1.mapped_to("2d-cs2"), cs1_to_cs2.map(pt_cs1))
    assert np.allclose(pt_cs1.mapped_to("2d-cs3"), cs2_to_cs3.map(cs1_to_cs2.map(pt_cs1)))
    assert np.allclose(pt_cs1.mapped_to("2d-cs3").mapped_to("2d-cs1"), pt_cs1)

    # inverses, too
    pt_cs3 = Point([0, 0], "2d-cs3")
    assert np.allclose(pt_cs3.mapped_to("2d-cs2"), cs2_to_cs3.inverse.map(pt_cs3))
    assert np.allclose(pt_cs3.mapped_to("2d-cs1"), cs1_to_cs2.inverse.map(cs2_to_cs3.inverse.map(pt_cs3)))


def test_coordinate_system_get_transform_to():
    """Test CoordinateSystem.get_transform_to method"""
    # Create transforms, which automatically creates coordinate systems in default graph
    cs1_to_cs2 = STTransform(scale=[2, 3], offset=[5, 10], from_cs="test_cs1", to_cs="test_cs2")
    cs2_to_cs3 = STTransform(scale=[1, 1], offset=[1, 1], from_cs="test_cs2", to_cs="test_cs3")
    
    # Get default graph and coordinate system objects
    default_graph = CoordinateSystemGraph.get_graph(None)
    cs1 = default_graph.system("test_cs1")
    cs2 = default_graph.system("test_cs2")
    cs3 = default_graph.system("test_cs3")
    
    # Test direct transform
    transform = cs1.get_transform_to(cs2)
    assert transform is cs1_to_cs2
    
    # Test inverse transform
    transform = cs2.get_transform_to(cs1)
    assert transform is cs1_to_cs2.inverse
    
    # Test pathfinding through intermediate coordinate system
    transform = cs1.get_transform_to(cs3)
    assert isinstance(transform, CompositeTransform)
    
    # Test transform from coordinate system to itself
    transform = cs1.get_transform_to(cs1)
    pt = Point([1, 2], cs1)
    mapped_pt = transform.map(pt)
    assert np.allclose(mapped_pt.coordinates, pt.coordinates)
    
    # Test with string coordinate system names
    transform = cs1.get_transform_to("test_cs2")
    assert transform is cs1_to_cs2
    
    # Test error when no path exists
    cs4 = default_graph.add_system("test_cs4", ndim=2)
    with raises(TypeError, match="No transform path from"):
        cs1.get_transform_to(cs4)


def test_coordinate_system_graph_transform():
    """Test CoordinateSystemGraph.transform method including pathfinding"""
    # Create transforms, which automatically creates coordinate systems in default graph
    cs1_to_cs2 = STTransform(scale=[2, 2, 2], offset=[1, 2, 3], from_cs="graph_cs1", to_cs="graph_cs2")
    cs2_to_cs3 = STTransform(scale=[0.5, 0.5, 0.5], offset=[0, 0, 0], from_cs="graph_cs2", to_cs="graph_cs3")
    cs3_to_cs4 = STTransform(scale=[1, 1, 1], offset=[10, 20, 30], from_cs="graph_cs3", to_cs="graph_cs4")
    
    # Get default graph and coordinate system objects
    default_graph = CoordinateSystemGraph.get_graph(None)
    cs1 = default_graph.system("graph_cs1")
    cs2 = default_graph.system("graph_cs2")
    cs3 = default_graph.system("graph_cs3")
    cs4 = default_graph.system("graph_cs4")
    
    # Test direct transform
    transform = default_graph.transform(cs1, cs2)
    assert transform is cs1_to_cs2
    
    # Test inverse transform
    transform = default_graph.transform(cs2, cs1)
    assert transform is cs1_to_cs2.inverse
    
    # Test pathfinding through one intermediate coordinate system
    transform = default_graph.transform(cs1, cs3)
    assert isinstance(transform, CompositeTransform)
    pt = Point([1, 2, 3], cs1)
    expected = cs2_to_cs3.map(cs1_to_cs2.map(pt))
    actual = transform.map(pt)
    assert np.allclose(actual.coordinates, expected.coordinates)
    assert actual.system is cs3
    
    # Test pathfinding through multiple intermediate coordinate systems
    transform = default_graph.transform(cs1, cs4)
    assert isinstance(transform, CompositeTransform)
    pt = Point([1, 2, 3], cs1)
    expected = cs3_to_cs4.map(cs2_to_cs3.map(cs1_to_cs2.map(pt)))
    actual = transform.map(pt)
    assert np.allclose(actual.coordinates, expected.coordinates)
    assert actual.system is cs4
    
    # Test identity transform (same coordinate system)
    transform = default_graph.transform(cs1, cs1)
    pt = Point([1, 2, 3], cs1)
    mapped_pt = transform.map(pt)
    assert np.allclose(mapped_pt.coordinates, pt.coordinates)
    assert mapped_pt.system is cs1
    
    # Test with string coordinate system names
    transform = default_graph.transform("graph_cs1", "graph_cs2")
    assert transform is cs1_to_cs2
    
    # Test mixed string and CoordinateSystem objects
    transform = default_graph.transform(cs1, "graph_cs2")
    assert transform is cs1_to_cs2
    transform = default_graph.transform("graph_cs1", cs2)
    assert transform is cs1_to_cs2
    
    # Test error when no path exists
    isolated_cs = default_graph.add_system("isolated_cs", ndim=3)
    with raises(TypeError, match="No transform path from"):
        default_graph.transform(cs1, isolated_cs)
    
    # Test error with nonexistent coordinate system
    with raises(NameError, match=missing_cs):
        default_graph.transform(cs1, "nonexistent_cs")


PARAMS = {
    "NullTransform": {},
    "TTransform": {"offset": (1, 1, 1)},
    "STTransform": {"scale": (2, 2, 2), "offset": (1, 2, 3)},
    "AffineTransform": {"matrix": [[0.5, 0, 0.707107], [0, 2, 0], [0.707107, 0, 0.5]], "offset": (4, 5, 6)},
    "SRT3DTransform": {"scale": (11, 11, 11), "angle": 45, "axis": (0, 1, 0), "offset": (1, 1, 1)},
    "TransposeTransform": {"axis_order": (1, 0, 2)},
    "LogTransform": {"base": (10, 10, 10)},
    "PolarTransform": {},
}


@pytest.mark.parametrize("type1", PARAMS.keys())
@pytest.mark.parametrize("type2", PARAMS.keys())
@pytest.mark.parametrize("inverse1", [False, True])
@pytest.mark.parametrize("inverse2", [False, True])
def test_transform_mapping(type1, type2, inverse1, inverse2):
    point = Point((1., 1., 1.), "cs1")
    if inverse1:
        cs2_from_cs1 = create_transform(type1, dims=(3, 3), systems=("cs2", "cs1"), **PARAMS[type1]).inverse
    else:
        cs2_from_cs1 = create_transform(type1, dims=(3, 3), systems=("cs1", "cs2"), **PARAMS[type1])
    if inverse2:
        cs3_from_cs2 = create_transform(type2, dims=(3, 3), systems=("cs3", "cs2"), **PARAMS[type2]).inverse
    else:
        cs3_from_cs2 = create_transform(type2, dims=(3, 3), systems=("cs2", "cs3"), **PARAMS[type2])

    assert str(cs2_from_cs1.map(point).system) == "cs2"

    explicitly_mapped = cs3_from_cs2.map(cs2_from_cs1.map(point))
    assert str(explicitly_mapped.system) == "cs3"

    with contextlib.suppress(NotImplementedError):  # ignore non-affine transforms here
        affine_mapped = cs3_from_cs2.as_affine().map(cs2_from_cs1.as_affine().map(point))
        assert str(affine_mapped.system) == "cs3"

    mult_mapped = (cs3_from_cs2 * cs2_from_cs1).map(point)
    assert str(mult_mapped.system) == "cs3"

    composite_mapped = CompositeTransform([cs2_from_cs1, cs3_from_cs2]).map(point)
    assert str(composite_mapped.system) == "cs3"

    with raises(TypeError, match=wrong_system):
        cs2_from_cs1.map(cs3_from_cs2.map(point))

    with raises(TypeError, match=mult_impossible):
        cs2_from_cs1 * cs3_from_cs2

    with raises(TypeError, match=comp_impossible):
        CompositeTransform([cs3_from_cs2, cs2_from_cs1])


def test_this_one_weird_situation():
    cs2_from_cs1 = create_transform("NullTransform", **{}, dims=(3, 3), systems=("cs1", "cs2"))
    cs3_from_cs2 = create_transform("SRT3DTransform", **PARAMS["SRT3DTransform"], dims=(3, 3), systems=("cs2", "cs3"))
    cs3_from_cs1 = cs3_from_cs2 * cs2_from_cs1
    assert str(cs3_from_cs1.map(Point([1, 1, 1], "cs1")).system) == "cs3"
    assert cs3_from_cs2.full_matrix.shape == (4, 4)  # just used to access it, really

    cs1_from_cs0 = create_transform("AffineTransform", **PARAMS["AffineTransform"], dims=(3, 3), systems=("cs0", "cs1"))
    cs3_from_cs0 = cs3_from_cs2 * cs2_from_cs1 * cs1_from_cs0
    assert str(cs3_from_cs0.map(Point([1, 1, 1], "cs0")).system) == "cs3"


@pytest.mark.parametrize("type1", PARAMS.keys())
@pytest.mark.parametrize("inverse1", [False, True])
@pytest.mark.parametrize("inverse2", [False, True])
def test_copy(type1, inverse1, inverse2):
    if inverse1:
        cs2_from_cs1 = create_transform(type1, **PARAMS[type1], dims=(3, 3), systems=("cs2", "cs1")).inverse
    else:
        cs2_from_cs1 = create_transform(type1, **PARAMS[type1], dims=(3, 3), systems=("cs1", "cs2"))
    copy = cs2_from_cs1.copy(from_cs="cs4")
    assert str(copy.systems[0]) == "cs4"
    assert str(copy.systems[1]) == "cs2"
    copy = cs2_from_cs1.copy(to_cs="cs5")
    assert str(copy.systems[0]) == "cs1"
    assert str(copy.systems[1]) == "cs5"
    copy = cs2_from_cs1.copy(from_cs="cs4", to_cs="cs5")
    assert str(copy.systems[0]) == "cs4"
    assert str(copy.systems[1]) == "cs5"


def test_composite_copy():
    cs2_from_cs1 = create_transform("AffineTransform", **PARAMS["AffineTransform"], dims=(3, 3), systems=("cs1", "cs2"))
    cs3_from_cs2 = create_transform("STTransform", **PARAMS["STTransform"], dims=(3, 3), systems=("cs2", "cs3"))
    cs3_from_cs1 = CompositeTransform([cs2_from_cs1, cs3_from_cs2])
    with pytest.raises(ValueError):
        cs3_from_cs1.copy(from_cs="cs4")
    with pytest.raises(ValueError):
        cs3_from_cs1.copy(to_cs="cs4")


@pytest.mark.parametrize("type1", PARAMS.keys())
def test_as_affine_systems(type1):
    xform = create_transform(type1, **PARAMS[type1], dims=(3, 3), systems=("affine1", "affine2"))
    point = Point([1, 2, 3], "affine1")
    with contextlib.suppress(NotImplementedError):  # ignore non-affine transforms here
        assert np.all(xform.as_affine().map(point) == xform.map(point))
        assert xform.as_affine().systems == xform.systems
        assert xform.inverse.as_affine().systems == xform.inverse.systems
        assert np.allclose(xform.inverse.as_affine().map(xform.map(point)), point)
        explicitly_looped = xform.inverse.as_affine().map(xform.as_affine().map(point))
        assert np.allclose(explicitly_looped, point)
        mult_loop = xform.inverse * xform
        assert np.allclose(mult_loop.as_affine().map(point), point)
        comp_loop = CompositeTransform([xform, xform.inverse])
        assert np.allclose(comp_loop.as_affine().map(point), point)


@pytest.mark.parametrize("type1", PARAMS.keys())
@pytest.mark.parametrize("inverse1", [False, True])
@pytest.mark.parametrize("inverse_composite", [False, True])
def test_composite_times_other(type1, inverse1, inverse_composite):
    pt_cs1 = Point([0., 0., 1.], "cs1")
    pt_cs3 = Point([0., 0., 0.], "cs3")
    if inverse_composite:
        cs1_from_cs2 = STTransform(scale=[3, 2, 1], offset=[10, 20, 30], from_cs="cs1", to_cs="cs2").inverse
        cs2_from_cs3 = STTransform(scale=[1, 1, 1], offset=[0, 0, -1], from_cs="cs3", to_cs="cs2")
        cs3_from_cs1 = CompositeTransform([cs2_from_cs3, cs1_from_cs2]).inverse
    else:
        cs2_from_cs1 = STTransform(scale=[3, 2, 1], offset=[10, 20, 30], from_cs="cs1", to_cs="cs2")
        cs3_from_cs2 = STTransform(scale=[1, 1, 1], offset=[0, 0, -1], from_cs="cs2", to_cs="cs3")
        cs3_from_cs1 = CompositeTransform([cs2_from_cs1, cs3_from_cs2])
    if inverse1:
        cs4_from_cs3 = create_transform(type1, **PARAMS[type1], dims=(3, 3), systems=("cs4", "cs3")).inverse
    else:
        cs4_from_cs3 = create_transform(type1, **PARAMS[type1], dims=(3, 3), systems=("cs3", "cs4"))

    assert str(cs3_from_cs1.map(pt_cs1).system) == "cs3"
    assert str(cs4_from_cs3.map(pt_cs3).system) == "cs4"

    explicitly_mapped = cs4_from_cs3.map(cs3_from_cs1.map(pt_cs1))
    assert str(explicitly_mapped.system) == "cs4"

    mult_mapped = (cs4_from_cs3 * cs3_from_cs1).map(pt_cs1)
    assert str(mult_mapped.system) == "cs4"

    composite_mapped = CompositeTransform([cs3_from_cs1, cs4_from_cs3]).map(pt_cs1)
    assert str(composite_mapped.system) == "cs4"

    with raises(TypeError, match=wrong_system):
        cs3_from_cs1.map(cs4_from_cs3.map(pt_cs1))

    if inverse_composite and (type1 != "NullTransform" or inverse1):
        comp_mult_err = mult_impossible
    else:
        comp_mult_err = comp_impossible
    with raises(TypeError, match=comp_mult_err):
        cs3_from_cs1 * cs4_from_cs3

    with raises(TypeError, match=comp_impossible):
        CompositeTransform([cs4_from_cs3, cs3_from_cs1])

    # check it works on the other side, too
    pt_cs0 = Point([0., 2., 0.], "cs0")
    cs0_to_cs1 = create_transform(type1, **PARAMS[type1], dims=(3, 3), systems=("cs0", "cs1"))

    assert str(cs0_to_cs1.map(pt_cs0).system) == "cs1"

    explicitly_mapped = cs3_from_cs1.map(cs0_to_cs1.map(pt_cs0))
    assert str(explicitly_mapped.system) == "cs3"

    mult_mapped = (cs3_from_cs1 * cs0_to_cs1).map(pt_cs0)
    assert str(mult_mapped.system) == "cs3"

    composite_mapped = CompositeTransform([cs0_to_cs1, cs3_from_cs1]).map(pt_cs0)
    assert str(composite_mapped.system) == "cs3"

    with raises(TypeError, match=wrong_system):
        cs0_to_cs1.map(cs3_from_cs1.map(pt_cs0))

    with raises(TypeError):
        cs0_to_cs1 * cs3_from_cs1

    with raises(TypeError):
        CompositeTransform([cs3_from_cs1, cs0_to_cs1])
