from collections import OrderedDict
from .converter import TransformConverter


def all_converters(superclass=None):
    if superclass is None:
        superclass = TransformConverter
    
    conv = OrderedDict()
    for cls in superclass.__subclasses__():
        conv[cls.name] = cls
        conv.update(all_converters(cls))
        
    return conv


_all_converters = None
def converter_class(name):
    global _all_converters
    if _all_converters is None or name not in _all_converters:
        _all_converters = all_converters()
        
    return _all_converters[name]


def make_converter(name, *args, **kwds):
    return converter_class(name)(*args, **kwds)


from . import _pyqtgraph, _vispy  # todo: ITK, Qt, scikit-image, matplotlib
