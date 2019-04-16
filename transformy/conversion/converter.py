

class TransformConverter(object):
    """Base for classes that implement conversion of transforms between different libraries.
    """
    name = None

    def convert_to(self, tr):
        """Return a transform converted to the target library, or raise an exception if this is not possible.
        
        The default implementation uses ``self._to_classes`` to look up the correct conversion function.
        """
        if self._import_error is not None:
            raise TypeError("Converting object of type %s to %s is not supported. (%s)" % (tr.__class__, self.name, self._import_error))
        if tr.__class__ not in self._to_classes:
            raise TypeError("Converting object of type %s to %s is not supported." % (tr.__class__, self.name))
        return self._to_classes[tr.__class__](tr)
        
    def convert_from(self, tr):
        """Return a transform converted from the target library, or raise an exception if this is not possible.
        
        The default implementation uses ``self._from_classes`` to look up the correct conversion function.
        """
        if self._import_error is not None:
            raise TypeError("Converting object of type %s from %s is not supported. (%s)" % (tr.__class__, self.name, self._import_error))
        if tr.__class__ not in self._from_classes:
            raise TypeError("Converting object of type %s from %s is not supported." % (tr.__class__, self.name))
        return self._from_classes[tr.__class__](tr)
