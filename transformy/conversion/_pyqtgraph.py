import numpy as np
from .converter import TransformConverter
from .. import linear


class PyqtgraphTransformConverter(TransformConverter):
    name = 'pyqtgraph'
    
    def __init__(self):
        try:
            import pyqtgraph
            self._import_error = None
        except ImportError as exc:
            self._import_error = str(exc)
            return
            
        self._to_classes = {
            linear.STTransform: self._STTransform_to_pg,
            linear.AffineTransform: self._AffineTransform_to_pg,
        }
        self._from_classes = {
            # pyqtgraph.SRTTransform: self._from_SRTTransform,
            # pyqtgraph.SRTTransform3D: self._from_SRTTransform,
            pyqtgraph.QtGui.QTransform: self._from_QTransform,
            pyqtgraph.QtGui.QMatrix4x4: self._from_QMatrix4x4,
            pyqtgraph.Transform3D: self._from_QMatrix4x4,
        }
    
    def _STTransform_to_pg(self, tr):
        import pyqtgraph
        if tr.dims == (2, 2):
            ptr = pyqtgraph.SRTTransform()
            ptr.setScale(tr.scale)
            ptr.setTranslate(tr.offset)
            return ptr
        elif tr.dims == (3, 3):
            ptr = pyqtgraph.SRTTransform3D()
            ptr.setScale(tr.scale)
            ptr.setTranslate(tr.offset)
            return ptr
        else:
            raise TypeError("Converting STTransform of dimension %r to pyqtgraph is not supported." % tr.dims)
    
    def _AffineTransform_to_pg(self, tr):
        import pyqtgraph
        if tr.dims == (2, 2):
            m = tr.matrix
            o = tr.offset
            ptr = pyqtgraph.QtGui.QTransform(m[0,0], m[1,0], 0.0, m[0,1], m[1,1], 0.0, o[0], o[1], 1.0)
            return ptr
        elif tr.dims == (3, 3):
            m = np.eye(4)
            m[:3, :3] = tr.matrix
            m[:3, 3] = tr.offset
            ptr = pyqtgraph.Transform3D(m)
            return ptr
        else:
            raise TypeError("Converting AffineTransform of dimension %r to pyqtgraph is not supported." % tr.dims)
    
    def _from_SRTTransform(self, tr):
        return linear.STTransform(offset=tr.getTranslation(), scale=tr.getScale())
        
    def _from_QTransform(self, tr):
        m = np.array([
            [tr.m11(), tr.m21()],
            [tr.m12(), tr.m22()],
        ])
        o = np.array([tr.m31(), tr.m32()])
        return linear.AffineTransform(matrix=m, offset=o)
        
    def _from_QMatrix4x4(self, tr):
        m = np.array(tr.copyDataTo()).reshape(4,4)
        return linear.AffineTransform(matrix=m[:3, :3], offset=m[:3, 3])
