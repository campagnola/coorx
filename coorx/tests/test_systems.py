import numpy as np
from pytest import raises
from coorx.coordinates import Point, PointArray
from coorx.systems import CoordinateSystemGraph
from coorx.linear import STTransform


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

    graph = CoordinateSystemGraph.get_graph()
    assert graph.transform('cs1', 'cs2') is cs1_to_cs2
    assert graph.transform('cs2', 'cs1') is cs1_to_cs2.inverse

    pt_cs2 = cs1_to_cs2.map(pt_cs1)

    with raises(TypeError, match=wrong_system):
        cs1_to_cs2.map(pt_cs2)
    with raises(TypeError, match=wrong_system):
        cs1_to_cs2.inverse.map(pt_cs1)

    assert pt_cs2.system is default_graph.system('cs2')
    assert cs1_to_cs2.imap(pt_cs2).system is default_graph.system('cs1')
    assert np.all(pt_cs2.coordinates == np.array([10, 20]))
    assert np.all(pt_cs1.coordinates == cs1_to_cs2.inverse.map(pt_cs2).coordinates)


