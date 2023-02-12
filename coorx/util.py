from .base_transform import BaseTransform

class AxisSelectionEmbeddedTransform(BaseTransform):
    def __init__(self, axes, transform, dims):
        super().__init__(dims=dims)
        self.axes = axes
        self.subtr = transform

    def _map(self, arr):
        out = arr.copy()
        out[:, self.axes] = self.subtr.map(arr[:, self.axes])
        return out

    def _imap(self, arr):
        out = arr.copy()
        out[:, axes] = self.subtr.imap(arr[:, axes])
        return out

