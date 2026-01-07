
import inspect
import threading
from typing import Callable
import weakref


class ChangeEvent:
    def __init__(self, transform, source_event=None):
        self.transform = transform
        self.source_event = source_event

    @property
    def sources(self):
        """A list of all transforms that changed leading to this event"""
        s = [self]
        if self.source_event is not None:
            s += self.source_event.sources
        return s


class CallbackRegistry:
    def __init__(self):
        # List of (is_weakref, callback or weakref) tuples
        self._callbacks: list[tuple[bool, Callable]] = []
        self.lock = threading.Lock()

    def add(self, cb, keep_reference):
        if keep_reference:
            cb_ref = (False, cb)
        else:
            weak_self = weakref.ref(self)

            def cleanup(dead_ref):
                registry = weak_self()
                if registry is not None:
                    registry.remove(dead_ref)

            if inspect.ismethod(cb):
                cb_ref = (True, weakref.WeakMethod(cb, cleanup))
            else:
                cb_ref = (True, weakref.ref(cb, cleanup))

        with self.lock:
            self._callbacks.append(cb_ref)

    def remove(self, cb):
        with self.lock:
            new_callbacks = []
            for is_ref, maybe_cb in self._callbacks:
                if is_ref:
                    cb_from_ref = maybe_cb()
                    if cb_from_ref is None:
                        # Clean up dead weak refs, too
                        continue
                    if cb_from_ref == cb:
                        continue
                else:
                    if maybe_cb == cb:
                        continue
                new_callbacks.append((is_ref, maybe_cb))
            self._callbacks = new_callbacks

    def __call__(self, *args, **kwargs):
        """Invoke all registered callbacks with the given arguments."""
        with self.lock:
            # Make a snapshot of callbacks to invoke
            callbacks = [cb_ref() if is_ref else cb_ref for is_ref, cb_ref in self._callbacks]
        for cb in callbacks:
            if cb is not None:
                cb(*args, **kwargs)
