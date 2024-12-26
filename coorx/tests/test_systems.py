import pickle

import numpy as np
import pytest
from coorx import CompositeTransform
from coorx import create_transform, Point
from coorx.coordinates import PointArray
from coorx.linear import STTransform, NullTransform
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
    pt_cs1 = Point([0, 0], "2d-cs1")
    parr_cs2 = PointArray([[1, 1], [1, 0], [0, 1], [1, 1]], "2d-cs2")

    assert pt_cs1.system is default_graph.system("2d-cs1")
    assert np.all(pt_cs1.coordinates == [0, 0])
    with raises(TypeError, match=missing_tr):
        pt_cs1.mapped_to("2d-cs2")
    with raises(NameError, match=missing_cs):
        pt_cs1.mapped_to("nonexistent_cs")
    with raises(TypeError, match=missing_tr):
        parr_cs2.mapped_to("2d-cs2")
    with raises(TypeError, match=wrong_ndim):
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


PARAMS = {
    "NullTransform": {},
    "TTransform": {"offset": (1, 1, 1)},
    "STTransform": {"scale": (2, 2, 2), "offset": (1, 2, 3)},
    "AffineTransform": {"matrix": [[0.5, 0, 0.707107], [0, 2, 0], [0.707107, 0, 0.5]], "offset": (4, 5, 6)},
    "SRT3DTransform": {"scale": (11, 11, 11), "angle": 45, "axis": (0, 1, 0), "offset": (1, 1, 1)},
    "LogTransform": {"base": (10, 10, 10)},
    "PolarTransform": {},
}


@pytest.mark.parametrize("type1", PARAMS.keys())
@pytest.mark.parametrize("type2", PARAMS.keys())
def test_transform_mapping(type1, type2):
    point = Point((1., 1., 1.), "cs1")
    transform1 = create_transform(type1, PARAMS[type1], dims=(3, 3), systems=("cs1", "cs2"))
    transform2 = create_transform(type2, PARAMS[type2], dims=(3, 3), systems=("cs2", "cs3"))

    assert str(transform1.map(point).system) == "cs2"

    explicitly_mapped = transform2.map(transform1.map(point))
    assert str(explicitly_mapped.system) == "cs3"

    mult_mapped = (transform2 * transform1).map(point)
    assert str(mult_mapped.system) == "cs3"

    composite_mapped = CompositeTransform(transform1, transform2).map(point)
    assert str(composite_mapped.system) == "cs3"

    with raises(TypeError, match=wrong_system):
        transform1.map(transform2.map(point))

    with raises(TypeError, match=mult_impossible):
        transform1 * transform2

    with raises(TypeError, match=comp_impossible):
        CompositeTransform(transform2, transform1)


@pytest.mark.parametrize("type1", PARAMS.keys())
def test_composite_times_other(type1):
    pt_cs1 = Point([0., 0., 1.], "cs1")
    pt_cs3 = Point([0., 0., 0.], "cs3")
    cs1_to_cs2 = STTransform(scale=[3, 2, 1], offset=[10, 20, 30], from_cs="cs1", to_cs="cs2")
    cs2_to_cs3 = STTransform(scale=[1, 1, 1], offset=[0, 0, -1], from_cs="cs2", to_cs="cs3")
    cs1_to_cs3 = CompositeTransform(cs1_to_cs2, cs2_to_cs3)
    cs3_to_cs4 = create_transform(type1, PARAMS[type1], dims=(3, 3), systems=("cs3", "cs4"))

    assert str(cs1_to_cs3.map(pt_cs1).system) == "cs3"
    assert str(cs3_to_cs4.map(pt_cs3).system) == "cs4"

    explicitly_mapped = cs3_to_cs4.map(cs1_to_cs3.map(pt_cs1))
    assert str(explicitly_mapped.system) == "cs4"

    mult_mapped = (cs3_to_cs4 * cs1_to_cs3).map(pt_cs1)
    assert str(mult_mapped.system) == "cs4"

    composite_mapped = CompositeTransform(cs1_to_cs3, cs3_to_cs4).map(pt_cs1)
    assert str(composite_mapped.system) == "cs4"

    with raises(TypeError, match=wrong_system):
        cs1_to_cs3.map(cs3_to_cs4.map(pt_cs1))

    with raises(TypeError, match=comp_impossible):
        cs1_to_cs3 * cs3_to_cs4

    with raises(TypeError, match=comp_impossible):
        CompositeTransform(cs3_to_cs4, cs1_to_cs3)

    # check it works on the other side, too
    pt_cs0 = Point([0., 2., 0.], "cs0")
    cs0_to_cs1 = create_transform(type1, PARAMS[type1], dims=(3, 3), systems=("cs0", "cs1"))

    assert str(cs0_to_cs1.map(pt_cs0).system) == "cs1"

    explicitly_mapped = cs1_to_cs3.map(cs0_to_cs1.map(pt_cs0))
    assert str(explicitly_mapped.system) == "cs3"

    mult_mapped = (cs1_to_cs3 * cs0_to_cs1).map(pt_cs0)
    assert str(mult_mapped.system) == "cs3"

    composite_mapped = CompositeTransform(cs0_to_cs1, cs1_to_cs3).map(pt_cs0)
    assert str(composite_mapped.system) == "cs3"

    with raises(TypeError, match=wrong_system):
        cs0_to_cs1.map(cs1_to_cs3.map(pt_cs0))

    with raises(TypeError, match=comp_impossible):
        cs0_to_cs1 * cs1_to_cs3

    with raises(TypeError, match=comp_impossible):
        CompositeTransform(cs1_to_cs3, cs0_to_cs1)
