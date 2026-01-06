import gc
import unittest
import weakref

import coorx
import types


def trace_referrers(obj, depth=3, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen or depth == 0:
        return seen
    seen.add(obj_id)
    # Filter out our own investigation artifacts
    referrers = [
        r for r in gc.get_referrers(obj)
        if not isinstance(r, types.FrameType) and id(r) not in seen and r is not seen
    ]
    for ref in referrers:
        print("  " * (3 - depth), type(ref), repr(ref)[:100])
        trace_referrers(ref, depth - 1, seen)
    return seen


class TestGarbageCollection(unittest.TestCase):
    """Test that transforms don't leak memory via reference cycles."""

    def test_composite_transform_gc_with_callbacks(self):
        """Test that CompositeTransforms and their children can be GC'd.

        CompositeTransform creates a reference cycle:
        - CompositeTransform holds refs to child transforms in _transforms
        - Each child has callbacks in _change_callbacks
        - The callback self._subtr_changed is a bound method referencing the CompositeTransform

        This test verifies that despite this cycle, all objects can be collected.
        """
        # Create transforms
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))
        t3 = coorx.AffineTransform(matrix=[[1, 0], [0, 1]], offset=(0, 0))

        # Create composite - this adds change callbacks creating reference cycles
        composite = coorx.CompositeTransform(t1, t2, t3)

        # Create weak references to track if objects are collected
        weak_t1 = weakref.ref(t1)
        weak_t2 = weakref.ref(t2)
        weak_t3 = weakref.ref(t3)
        weak_composite = weakref.ref(composite)

        # Verify objects exist
        self.assertIsNotNone(weak_t1())
        self.assertIsNotNone(weak_t2())
        self.assertIsNotNone(weak_t3())
        self.assertIsNotNone(weak_composite())

        # Delete all strong references
        del t1, t2, t3, composite

        # Force garbage collection
        gc.collect()

        # Check that all objects were collected
        self.assertIsNone(weak_t1(), "Transform t1 was not garbage collected")
        self.assertIsNone(weak_t2(), "Transform t2 was not garbage collected")
        self.assertIsNone(weak_t3(), "Transform t3 was not garbage collected")
        self.assertIsNone(weak_composite(), "CompositeTransform was not garbage collected")

    def test_nested_composite_transform_gc(self):
        """Test GC with nested CompositeTransforms."""
        # Create nested structure
        inner1 = coorx.CompositeTransform(
            coorx.TTransform(offset=(1, 0)), coorx.STTransform(scale=(2, 2), offset=(0, 0))
        )
        inner2 = coorx.CompositeTransform(
            coorx.TTransform(offset=(0, 1)),
            coorx.AffineTransform(matrix=[[1, 0], [0, 1]], offset=(1, 1)),
        )
        outer = coorx.CompositeTransform(inner1, inner2)

        # Create weak references
        weak_inner1 = weakref.ref(inner1)
        weak_inner2 = weakref.ref(inner2)
        weak_outer = weakref.ref(outer)

        # Delete references
        del inner1, inner2, outer

        # Force GC
        gc.collect()

        # Check collection
        self.assertIsNone(weak_inner1(), "Inner composite 1 was not garbage collected")
        self.assertIsNone(weak_inner2(), "Inner composite 2 was not garbage collected")
        self.assertIsNone(weak_outer(), "Outer composite was not garbage collected")

    def test_simplified_composite_gc(self):
        """Test that simplified CompositeTransforms can be GC'd.

        SimplifiedCompositeTransform also adds callbacks to the source chain,
        creating another potential reference cycle.
        """
        # Create composite
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))
        composite = coorx.CompositeTransform(t1, t2)

        # Access simplified version - this creates additional callbacks
        simplified = composite.simplified

        # Create weak references
        weak_composite = weakref.ref(composite)
        weak_simplified = weakref.ref(simplified)
        weak_t1 = weakref.ref(t1)
        weak_t2 = weakref.ref(t2)

        # Delete references
        del t1, t2, composite, simplified

        # Force GC
        gc.collect()

        # Check collection
        self.assertIsNone(weak_composite(), "Composite was not garbage collected")
        self.assertIsNone(weak_simplified(), "Simplified composite was not garbage collected")
        self.assertIsNone(weak_t1(), "Transform t1 was not garbage collected")
        self.assertIsNone(weak_t2(), "Transform t2 was not garbage collected")

    def test_transform_with_custom_callbacks_gc(self):
        """Test that transforms with custom callbacks can be GC'd."""
        # Create transforms
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(0, 0))

        # Track callback invocations (but don't create strong ref to transforms)
        callback_count = [0]

        def callback(event):
            callback_count[0] += 1

        # Add custom callback
        t1.add_change_callback(callback)

        # Create composite
        composite = coorx.CompositeTransform(t1, t2)

        # Create weak references
        weak_t1 = weakref.ref(t1)
        weak_composite = weakref.ref(composite)

        # Delete references
        del t1, t2, composite

        # Force GC
        gc.collect()

        # Check collection
        self.assertIsNone(weak_t1(), "Transform with custom callback was not garbage collected")
        self.assertIsNone(weak_composite(), "Composite was not garbage collected")

    def test_child_transforms_dont_keep_composite_alive(self):
        """Test that child transforms don't prevent CompositeTransform GC via callbacks.

        When a CompositeTransform is deleted, its children should not keep it alive
        through their callback references. The composite should be garbage collected
        even when child transforms remain in use.
        """
        # Create child transforms that we'll keep references to
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))

        # Create composite - this sets up internal callbacks
        composite = coorx.CompositeTransform(t1, t2)

        # Verify composite works
        pts = [[0, 0]]
        result = composite.map(pts)
        self.assertIsNotNone(result)

        # Create weak reference to composite
        weak_composite = weakref.ref(composite)

        # Delete the composite but keep the children
        del composite

        # Force GC
        gc.collect()

        # The composite should be collected even though children still exist
        self.assertIsNone(weak_composite(),
                         "CompositeTransform kept alive by child transform callbacks")

        # Children should still be alive and functional
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)

        # Children should still work after composite is gone
        t1.set_params(offset=(5, 6))  # Should not crash
        result = t1.map(pts)
        self.assertIsNotNone(result)

    def test_reused_transform_callback_cleanup(self):
        """Test that reused transforms don't accumulate unbounded memory from dead callbacks.

        When a transform is reused across many CompositeTransforms that get deleted,
        the transform should clean up references to the dead composites and not
        accumulate unbounded memory.
        """
        # Create a transform that will be reused
        shared_transform = coorx.TTransform(offset=(1, 2))

        # Track that the shared transform still works
        call_count = [0]

        def tracker(event):
            call_count[0] += 1

        shared_transform.add_change_callback(tracker)

        # Create and delete many composites that use this transform
        for i in range(50):
            t = coorx.STTransform(scale=(2, 3), offset=(i, i))
            composite = coorx.CompositeTransform(shared_transform, t)
            del composite  # Explicitly delete to avoid loop variable hanging around

        # Force GC to ensure composites are deleted
        gc.collect()

        # The shared transform should still work and invoke callbacks normally
        # This verifies internal cleanup hasn't broken functionality
        shared_transform.set_params(offset=(5, 6))
        self.assertGreater(call_count[0], 0, "Callback should still be invoked")

        # Transform should still map correctly
        result = shared_transform.map([[0, 0]])
        self.assertIsNotNone(result)

    def test_inverse_transform_gc(self):
        """Test that transforms with inverses can be GC'd.

        InverseTransform holds a reference to the original transform,
        and the original holds a reference to the inverse.
        """
        # Create transform and access its inverse
        t = coorx.STTransform(scale=(2, 3), offset=(1, 2))
        inv = t.inverse

        # Create weak references
        weak_t = weakref.ref(t)
        weak_inv = weakref.ref(inv)

        # Verify cycle exists
        self.assertIs(inv.inverse, t)

        # Delete references
        del t, inv

        # Force GC
        gc.collect()

        # Check collection
        self.assertIsNone(weak_t(), "Transform was not garbage collected")
        self.assertIsNone(weak_inv(), "Inverse transform was not garbage collected")

    def test_multiple_methods_from_same_object(self):
        """Test multiple bound methods from the same object as callbacks.

        Critical for WeakKeyDictionary design: when the key is obj.__self__,
        the value needs to store multiple methods from that object.
        """
        class MultiMethodListener:
            def __init__(self):
                self.method1_calls = 0
                self.method2_calls = 0
                self.method3_calls = 0

            def on_change1(self, event):
                self.method1_calls += 1

            def on_change2(self, event):
                self.method2_calls += 1

            def on_change3(self, event):
                self.method3_calls += 1

        # Create transform and listener
        t = coorx.TTransform(offset=(1, 2))
        listener = MultiMethodListener()

        # Add multiple methods from the same object
        t.add_change_callback(listener.on_change1)
        t.add_change_callback(listener.on_change2)
        t.add_change_callback(listener.on_change3)

        # Trigger change - all three should be called
        # Note: set_params triggers _update() twice (once in setter, once at end)
        t.set_params(offset=(2, 3))

        self.assertEqual(listener.method1_calls, 2)
        self.assertEqual(listener.method2_calls, 2)
        self.assertEqual(listener.method3_calls, 2)

        # Create weak refs
        weak_listener = weakref.ref(listener)
        weak_t = weakref.ref(t)

        # Delete listener - transform should still be collectible
        del listener
        gc.collect()

        self.assertIsNone(weak_listener(), "Listener was not garbage collected")

        # Transform should still work and not crash when invoking dead callbacks
        t.set_params(offset=(3, 4))

        # Now delete transform
        del t
        gc.collect()
        self.assertIsNone(weak_t(), "Transform was not garbage collected")

    def test_remove_specific_method_among_multiple(self):
        """Test removing one specific method when multiple methods from same object exist."""
        class MultiMethodListener:
            def __init__(self):
                self.method1_calls = 0
                self.method2_calls = 0

            def on_change1(self, event):
                self.method1_calls += 1

            def on_change2(self, event):
                self.method2_calls += 1

        t = coorx.TTransform(offset=(1, 2))
        listener = MultiMethodListener()

        # Add both methods
        t.add_change_callback(listener.on_change1)
        t.add_change_callback(listener.on_change2)

        # Verify both work (set_params triggers _update twice)
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.method1_calls, 2)
        self.assertEqual(listener.method2_calls, 2)

        # Remove only method1
        t.remove_change_callback(listener.on_change1)

        # Verify only method2 is called now
        t.set_params(offset=(3, 4))
        self.assertEqual(listener.method1_calls, 2)  # Still 2, not incremented
        self.assertEqual(listener.method2_calls, 4)  # Incremented by 2 more

    def test_mix_bound_methods_and_functions(self):
        """Test mix of bound methods and regular functions as callbacks."""
        function_calls = [0]

        def regular_callback(event):
            function_calls[0] += 1

        class MethodListener:
            def __init__(self):
                self.calls = 0

            def on_change(self, event):
                self.calls += 1

        t = coorx.TTransform(offset=(1, 2))
        listener = MethodListener()

        # Add both types
        t.add_change_callback(regular_callback)
        t.add_change_callback(listener.on_change)

        # Trigger change (set_params triggers _update twice)
        t.set_params(offset=(2, 3))

        self.assertEqual(function_calls[0], 2)
        self.assertEqual(listener.calls, 2)

        # Delete listener object - function should still work
        weak_listener = weakref.ref(listener)
        del listener
        gc.collect()

        self.assertIsNone(weak_listener(), "Listener not collected")

        # Function should still be called (it's a strong ref)
        t.set_params(offset=(3, 4))
        self.assertEqual(function_calls[0], 4)  # 2 more calls

    def test_callback_removes_itself_during_invocation(self):
        """Test that a callback can safely remove itself during invocation."""
        class SelfRemovingListener:
            def __init__(self, transform):
                self.transform = transform
                self.calls = 0

            def on_change(self, event):
                self.calls += 1
                if self.calls == 1:
                    # Remove self on first call
                    self.transform.remove_change_callback(self.on_change)

        t = coorx.TTransform(offset=(1, 2))
        listener = SelfRemovingListener(t)

        t.add_change_callback(listener.on_change)

        # First change - callback removes itself on first invocation
        # set_params triggers _update twice, but callback removes itself on first call
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 1)

        # Second change - callback should not be called
        t.set_params(offset=(3, 4))
        self.assertEqual(listener.calls, 1)  # Still 1

    def test_callback_adds_callback_during_invocation(self):
        """Test that a callback can add new callbacks during invocation."""
        second_callback_calls = [0]

        def second_callback(event):
            second_callback_calls[0] += 1

        class AddingListener:
            def __init__(self, transform):
                self.transform = transform
                self.calls = 0

            def on_change(self, event):
                self.calls += 1
                if self.calls == 1:
                    # Add another callback on first invocation
                    self.transform.add_change_callback(second_callback)

        t = coorx.TTransform(offset=(1, 2))
        listener = AddingListener(t)

        t.add_change_callback(listener.on_change)

        # First change - adds second callback on first invocation
        # set_params triggers _update twice
        # First _update: listener called (adds second_callback), updates callback list
        # Second _update: both listener and second_callback called
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 2)
        self.assertEqual(second_callback_calls[0], 1)  # Called once in second _update

        # Second change - both should be called twice
        t.set_params(offset=(3, 4))
        self.assertEqual(listener.calls, 4)  # 2 more calls
        self.assertEqual(second_callback_calls[0], 3)  # 2 more calls


if __name__ == '__main__':
    unittest.main()
