import pickle
import numpy as np
from coorx import CompositeTransform
from pytest import raises
from coorx.coordinates import Point, PointArray
from coorx.systems import CoordinateSystemGraph
from coorx.linear import STTransform, NullTransform

missing_tr = "No transform defined linking"
missing_cs = "No coordinate system named"
wrong_ndim = r"is \dD \(expected \dD\)"
wrong_system = "maps from system"


def test_coordinate_systems():
    default_graph = CoordinateSystemGraph.get_graph(None)
    pt_cs1 = Point([0, 0], "cs1")
    parr_cs2 = PointArray([[1, 1], [1, 0], [0, 1], [1, 1]], 'cs2')

    assert pt_cs1.system is default_graph.system('cs1')
    assert np.all(pt_cs1.coordinates == [0, 0])
    with raises(TypeError, match=missing_tr):
        pt_cs1.mapped_to('cs2')
    with raises(NameError, match=missing_cs):
        pt_cs1.mapped_to('nonexistent_cs')
    with raises(TypeError, match=missing_tr):
        parr_cs2.mapped_to('cs2')
    with raises(TypeError, match=wrong_ndim):
        Point([0, 0, 0], 'cs1')  # wrong ndim
 
    cs1_to_cs2 = STTransform(scale=[3, 2], offset=[10, 20], from_cs='cs1', to_cs='cs2')

    assert default_graph.transform('cs1', 'cs2') is cs1_to_cs2
    assert default_graph.transform('cs2', 'cs1') is cs1_to_cs2.inverse

    pt_cs2 = cs1_to_cs2.map(pt_cs1)

    with raises(TypeError, match=wrong_system):
        cs1_to_cs2.map(pt_cs2)
    with raises(TypeError, match=wrong_system):
        cs1_to_cs2.inverse.map(pt_cs1)

    assert pt_cs2.system is default_graph.system('cs2')
    assert cs1_to_cs2.imap(pt_cs2).system is default_graph.system('cs1')
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


def test_composite_times_null():
    cs1_to_cs2 = STTransform(scale=[3, 2], offset=[10, 20], from_cs='cs1', to_cs='cs2')
    cs2_to_cs3 = STTransform(scale=[1, 1], offset=[0, 0], from_cs='cs2', to_cs='cs3')
    null_cs3_to_cs4 = NullTransform(2, from_cs='cs3', to_cs='cs4')

    comp = CompositeTransform([cs1_to_cs2, cs2_to_cs3])
    mult = null_cs3_to_cs4 * comp
    pt_cs1 = Point([0, 0], 'cs1')
    pt_cs4 = mult.map(pt_cs1)
    assert pt_cs4.system == 'cs4'
