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

        This is the actual leak: when a CompositeTransform is deleted, its children
        still have callbacks (self._subtr_changed bound methods) that reference the
        composite, keeping it alive.
        """
        # Create child transforms that we'll keep references to
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))

        # Create composite - this adds callbacks to t1 and t2
        composite = coorx.CompositeTransform(t1, t2)

        # Verify callbacks were added
        self.assertGreater(len(t1._change_callbacks), 0, "No callbacks added to t1")
        self.assertGreater(len(t2._change_callbacks), 0, "No callbacks added to t2")

        # Create weak reference to composite
        weak_composite = weakref.ref(composite)

        # Delete the composite but keep the children
        del composite

        # Force GC
        gc.collect()

        # The composite should be collected even though children still exist
        self.assertIsNone(weak_composite(),
                         "CompositeTransform kept alive by child transform callbacks")

        # Children should still be alive
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)

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


if __name__ == '__main__':
    unittest.main()
