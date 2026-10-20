
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


def _callback_key(cb):
    """Return a hashable identity for *cb* that holds no strong reference to it.

    Accessing a bound method builds a fresh method object every time, so bound
    methods are keyed by the identity of their object together with the underlying
    function. That is exactly how CPython compares two bound methods for equality.
    Any other callable is keyed by its own identity.

    The key deliberately excludes the callback itself, so that a weakly registered
    callback remains collectable while it is registered.
    """
    if inspect.ismethod(cb):
        return id(cb.__self__), cb.__func__
    return id(cb), None


class CallbackRegistry:
    """A thread-safe registry for callbacks, supporting weak references.

    Every operation that names a single callback -- add, remove, and the cleanup
    that runs when a weakly referenced callback is collected -- costs O(1). A
    transform observed by many subscribers therefore does not get slower as those
    subscribers come and go.

    Callbacks are identified by identity rather than by equality; see
    :func:`_callback_key`.
    """
    def __init__(self):
        # Maps {callback key: [(is_weakref, callback or weakref), ...]}.
        # Keys are held in registration order. A key maps to more than one entry
        # only when the same callback was registered with duplicates='allow'.
        self._callbacks: dict[tuple, list[tuple[bool, Callable]]] = {}
        self.lock = threading.RLock()  # Use reentrant lock to allow add() to call remove()

    def _real_callbacks(self):
        """Return all live callbacks, in registration order."""
        return [
            cb
            for entries in self._callbacks.values()
            for is_weakref, cb_ref in entries
            if (cb := (cb_ref() if is_weakref else cb_ref)) is not None
        ]

    def _live_entries(self, key):
        """Return the (is_weakref, cb_ref) entries under *key* whose callback is alive."""
        return [
            (is_weakref, cb_ref)
            for is_weakref, cb_ref in self._callbacks.get(key, ())
            if (cb_ref() if is_weakref else cb_ref) is not None
        ]

    def _discard_dead(self, key, dead_ref):
        """Drop the entry holding *dead_ref*; called when a weak callback is collected.

        Only the entry holding this exact weakref is dropped. CPython reuses the
        addresses of freed objects, so by the time this runs a different callback
        may already have claimed the same key.
        """
        with self.lock:
            entries = self._callbacks.get(key)
            if entries is None:
                return
            remaining = [entry for entry in entries if entry[1] is not dead_ref]
            if remaining:
                self._callbacks[key] = remaining
            else:
                del self._callbacks[key]

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
        duplicates : {'error', 'ignore', 'allow'}
            How to handle duplicate registrations of the same callback.
            'error' raises a ValueError, 'ignore' does nothing, 'allow' registers
            the callback multiple times.
        """
        key = _callback_key(cb)
        if keep_reference:
            cb_ref = (False, cb)
        else:
            # Create a cleanup call back so dead weakrefs are removed from the registry
            weak_self = weakref.ref(self)
            def cleanup(dead_ref):
                registry = weak_self()
                if registry is not None:
                    registry._discard_dead(key, dead_ref)

            # create weak reference to the callback
            if inspect.ismethod(cb):
                cb_ref = (True, weakref.WeakMethod(cb, cleanup))
            else:
                cb_ref = (True, weakref.ref(cb, cleanup))

        with self.lock:
            # Prevent duplicate registrations
            if duplicates == 'allow':
                # Add the new callback regardless of previous registrations
                self._callbacks.setdefault(key, []).append(cb_ref)
                return

            # check if we have registered this callback already
            already_registered = self._live_entries(key)
            # if already registered, handle according to duplicates policy
            if already_registered:
                if duplicates == 'error':
                    raise ValueError("Callback already registered")
                elif duplicates == 'ignore':
                    is_weakref = already_registered[0][0]
                    if keep_reference and is_weakref:
                        # Upgrade weak reference to strong reference
                        self.remove(cb)
                        self._callbacks.setdefault(key, []).append(cb_ref)
                    else:
                        # just ignore
                        return
            else:
                self._callbacks.setdefault(key, []).append(cb_ref)

    def remove(self, cb):
        """Unregister every registration of *cb*. Does nothing if it is not registered."""
        with self.lock:
            self._callbacks.pop(_callback_key(cb), None)

    def __call__(self, *args, **kwargs):
        """Invoke all registered callbacks with the given arguments."""
        with self.lock:
            # Make a snapshot of callbacks to invoke
            callbacks = self._real_callbacks()
        for cb in callbacks:
            cb(*args, **kwargs)
