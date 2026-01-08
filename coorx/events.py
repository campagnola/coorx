
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
    """A thread-safe registry for callbacks, supporting weak references."""
    def __init__(self):
        # List of (is_weakref, callback or weakref) tuples
        self._callbacks: list[tuple[bool, Callable]] = []
        self.lock = threading.Lock()

    def _real_callbacks(self):
        """Return (callback, is_weakref, maybe_weakref) tuples for all live callbacks."""
        return [(cb, is_weakref, cb_ref) for is_weakref, cb_ref in self._callbacks if (cb := (cb_ref() if is_weakref else cb_ref)) is not None]

    def add(self, cb, keep_reference, duplicates='error'):
        """Register a callback.
        If keep_reference is False, a weak reference to the callback is stored,
        allowing it to be garbage collected if there are no other references.

        Parameters
        ----------
        cb : Callable
            The callback to register.
        keep_reference : bool
            Whether to keep a strong reference to the callback.
        duplicates : {'error', 'ignore', 'add'}
            How to handle duplicate registrations of the same callback.
            'error' raises a ValueError, 'ignore' does nothing, 'add' registers
            the callback multiple times.
        """
        if keep_reference:
            cb_ref = (False, cb)
        else:
            # Create a cleanup call back so dead weakrefs are removed from the registry
            weak_self = weakref.ref(self)
            def cleanup(dead_ref):
                registry = weak_self()
                if registry is not None:
                    registry.remove(dead_ref)

            # create weak reference to the callback
            if inspect.ismethod(cb):
                cb_ref = (True, weakref.WeakMethod(cb, cleanup))
            else:
                cb_ref = (True, weakref.ref(cb, cleanup))

        with self.lock:
            # Prevent duplicate registrations
            if duplicates == 'allow':
                # Add the new callback regardless of previous registrations
                self._callbacks.append(cb_ref)
            else:
                # check if we have registered this callback already
                already_registered = False
                is_weakref = False
                for other_cb, is_weakref, _ in self._real_callbacks():
                    if other_cb == cb:
                        already_registered = True
                        break
                # if already registered, handle according to duplicates policy
                if already_registered:
                    if duplicates == 'error':
                        raise ValueError("Callback already registered")
                    elif duplicates == 'ignore':
                        if keep_reference and is_weakref:
                            # Upgrade weak reference to strong reference
                            self.remove(cb)
                            self._callbacks.append(cb_ref)
                        else:
                            # just ignore
                            return
                else:
                    self._callbacks.append(cb_ref)


    def remove(self, cb):
        with self.lock:
            new_callbacks = []
            for other_cb, is_ref, maybe_cb in self._real_callbacks():
                if other_cb == cb:
                    continue
                new_callbacks.append((is_ref, maybe_cb))
            self._callbacks = new_callbacks

    def __call__(self, *args, **kwargs):
        """Invoke all registered callbacks with the given arguments."""
        with self.lock:
            # Make a snapshot of callbacks to invoke
            callbacks = self._real_callbacks()
        for cb, _, _ in callbacks:
            if cb is not None:
                cb(*args, **kwargs)
