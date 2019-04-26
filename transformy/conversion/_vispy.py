import numpy as np
from .converter import TransformConverter
from .. import linear


class VispyTransformConverter(TransformConverter):
    name = 'vispy'
    
    def __init__(self):
        try:
            import vispy.scene
            self._import_error = None
        except ImportError as exc:
            self._import_error = str(exc)
            return
            
        # By some strange luck, the vispy transforms have the same name! Hmmm.
        self._to_classes = {
            linear.STTransform: self._to_STTransform,
            linear.NullTransform: self._to_NullTransform,
            linear.AffineTransform: self._to_MatrixTransform,
        }
        self._from_classes = {
            vispy.scene.NullTransform: self._from_NullTransform,
            vispy.scene.STTransform: self._from_STTransform,
            # vispy.scene.MatrixTransform: self._from_MatrixTransform,
        }
    
    def _to_NullTransform(self, tr):
        import vispy.scene
        return vispy.scene.NullTransform()

    def _from_NullTransform(self, tr):
        return linear.NullTransform()
    
    def _to_STTransform(self, tr):
        import vispy.scene
        return vispy.scene.STTransform(translate=tr.offset, scale=tr.scale)
    
    def _from_STTransform(self, tr):
        return linear.STTransform(offset=tr.translate[:3], scale=tr.scale[:3])
        
    def _to_MatrixTransform(self, tr):
        import vispy.scene
        m = np.eye(4)
        m[:3, :3] = tr.matrix
        m[:3, 3] = tr.offset
        return vispy.scene.MatrixTransform(m.T)
    
    # def _from_AffineTransform(self, tr):
    #     return linear.STTransform(offset=tr.translate, scale=tr.scale)
        
