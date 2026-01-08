"""Test garbage collection behavior with weak/strong callback references."""

import gc
import unittest
import weakref

import coorx


class TestCallbackReferenceSemantics(unittest.TestCase):
    """Test that keep_reference parameter correctly controls GC behavior."""

    def test_bound_method_weak_reference_default(self):
        """By default, bound method callbacks should use weak references."""

        class Listener:
            def __init__(self):
                self.calls = 0

            def on_change(self, event):
                self.calls += 1

        t = coorx.TTransform(offset=(1, 2))
        listener = Listener()

        # Add callback without keep_reference (defaults to False/weak)
        t.add_change_callback(listener.on_change)

        # Verify callback works
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 1)

        # Delete listener - callback should stop being invoked
        weak_listener = weakref.ref(listener)
        del listener
        gc.collect()

        self.assertIsNone(weak_listener(), "Listener should be garbage collected")

        # Trigger change - should not crash, just skip dead callback
        t.set_params(offset=(3, 4))  # Should not crash

    def test_bound_method_strong_reference_explicit(self):
        """With keep_reference=True, bound methods should be kept alive."""

        class Listener:
            def __init__(self):
                self.calls = 0

            def on_change(self, event):
                self.calls += 1

        t = coorx.TTransform(offset=(1, 2))
        listener = Listener()

        # Add callback with keep_reference=True
        t.add_change_callback(listener.on_change, keep_reference=True)

        # Verify callback works
        self.assertEqual(listener.calls, 0)
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 1)

        # Delete listener - but transform still holds reference
        weak_listener = weakref.ref(listener)
        del listener
        gc.collect()

        # Listener should NOT be collected (transform keeps it alive)
        self.assertIsNotNone(weak_listener(), "Listener should be kept alive by transform")

        # Callback should still work
        t.set_params(offset=(3, 4))
        self.assertEqual(weak_listener().calls, 2)

    def test_function_requires_keep_reference(self):
        """Regular functions must use keep_reference=True."""
        call_count = [0]

        def callback(event):
            call_count[0] += 1

        t = coorx.TTransform(offset=(1, 2))

        # Add function with keep_reference=True
        t.add_change_callback(callback, keep_reference=True)

        # Verify callback works
        t.set_params(offset=(2, 3))
        self.assertEqual(call_count[0], 1)

        # Function should keep working (strong ref)
        t.set_params(offset=(3, 4))
        self.assertEqual(call_count[0], 2)

    def test_composite_transform_gc_with_weak_callbacks(self):
        """CompositeTransform should be collectible when using weak callbacks."""
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))

        composite = coorx.CompositeTransform([t1, t2])

        # Verify composite works
        result = composite.map([[0, 0]])
        self.assertIsNotNone(result)

        # Create weak references
        weak_composite = weakref.ref(composite)
        weak_t1 = weakref.ref(t1)
        weak_t2 = weakref.ref(t2)

        # Delete all references
        del t1, t2, composite
        gc.collect()

        # All should be collected
        self.assertIsNone(weak_composite(), "Composite should be collected")
        self.assertIsNone(weak_t1(), "Child transform t1 should be collected")
        self.assertIsNone(weak_t2(), "Child transform t2 should be collected")

    def test_child_transforms_dont_keep_composite_alive(self):
        """Child transforms with weak callbacks don't keep composite alive."""
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))

        composite = coorx.CompositeTransform([t1, t2])

        # Verify composite works
        result = composite.map([[0, 0]])
        self.assertIsNotNone(result)

        weak_composite = weakref.ref(composite)

        # Delete composite but keep children
        del composite
        gc.collect()

        # Composite should be collected even though children exist
        self.assertIsNone(weak_composite(), "Composite should be collected")

        # Children should still work
        t1.set_params(offset=(5, 6))
        result = t1.map([[0, 0]])
        self.assertIsNotNone(result)

    def test_multiple_weak_methods_from_same_object(self):
        """Multiple weak method callbacks from same object all work correctly."""

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

        t = coorx.TTransform(offset=(1, 2))
        listener = MultiMethodListener()

        # Add multiple methods from same object (weak refs)
        t.add_change_callback(listener.on_change1)
        t.add_change_callback(listener.on_change2)
        t.add_change_callback(listener.on_change3)

        # All three should be called
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.method1_calls, 1)
        self.assertEqual(listener.method2_calls, 1)
        self.assertEqual(listener.method3_calls, 1)

        # Delete listener - all callbacks should stop
        weak_listener = weakref.ref(listener)
        del listener
        gc.collect()

        self.assertIsNone(weak_listener(), "Listener should be collected")

        # Should not crash
        t.set_params(offset=(3, 4))

    def test_remove_specific_weak_method_among_multiple(self):
        """Removing one weak method callback leaves others working."""

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

        # Add both methods (weak refs)
        t.add_change_callback(listener.on_change1)
        t.add_change_callback(listener.on_change2)

        # Verify both work
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.method1_calls, 1)
        self.assertEqual(listener.method2_calls, 1)

        # Remove only method1
        t.remove_change_callback(listener.on_change1)

        # Only method2 should be called
        t.set_params(offset=(3, 4))
        self.assertEqual(listener.method1_calls, 1)  # No change
        self.assertEqual(listener.method2_calls, 2)  # Incremented

    def test_mix_weak_and_strong_callbacks(self):
        """Mix of weak (method) and strong (function) callbacks work together."""
        function_calls = [0]

        def strong_callback(event):
            function_calls[0] += 1

        class WeakListener:
            def __init__(self):
                self.calls = 0

            def on_change(self, event):
                self.calls += 1

        t = coorx.TTransform(offset=(1, 2))
        listener = WeakListener()

        # Add strong (function) and weak (method) callbacks
        t.add_change_callback(strong_callback, keep_reference=True)
        t.add_change_callback(listener.on_change)  # weak by default

        # Both should be called
        t.set_params(offset=(2, 3))
        self.assertEqual(function_calls[0], 1)
        self.assertEqual(listener.calls, 1)

        # Delete listener - only function should continue
        weak_listener = weakref.ref(listener)
        del listener
        gc.collect()

        self.assertIsNone(weak_listener(), "Weak listener should be collected")

        # Only strong callback should be called
        t.set_params(offset=(3, 4))
        self.assertEqual(function_calls[0], 2)  # Still called

    def test_strong_method_callback_keeps_object_alive(self):
        """Strong method callbacks keep their object alive."""

        class Listener:
            def __init__(self):
                self.calls = 0

            def on_change(self, event):
                self.calls += 1

        t = coorx.TTransform(offset=(1, 2))
        listener = Listener()

        # Add with keep_reference=True (strong)
        t.add_change_callback(listener.on_change, keep_reference=True)

        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 1)

        # Delete listener reference
        weak_listener = weakref.ref(listener)
        del listener
        gc.collect()

        # Should NOT be collected (transform keeps it alive)
        self.assertIsNotNone(weak_listener(), "Strong callback should keep listener alive")

        # Should still be called
        t.set_params(offset=(3, 4))
        self.assertEqual(weak_listener().calls, 2)

    def test_reused_transform_with_many_weak_callbacks(self):
        """Reused transform handles many weak callback additions/deletions."""
        shared_transform = coorx.TTransform(offset=(1, 2))

        # Track that shared transform still works
        call_count = [0]

        def tracker(event):
            call_count[0] += 1

        shared_transform.add_change_callback(tracker, keep_reference=True)

        # Create and delete many composites
        for i in range(50):
            t = coorx.STTransform(scale=(2, 3), offset=(i, i))
            composite = coorx.CompositeTransform([shared_transform, t])
            del composite

        gc.collect()

        # Transform should still work
        shared_transform.set_params(offset=(5, 6))
        self.assertEqual(call_count[0], 1, "Callback should still work")

        # Transform should still map
        result = shared_transform.map([[0, 0]])
        self.assertIsNotNone(result)

    def test_weak_callback_removes_itself_during_invocation(self):
        """Weak callback can safely remove itself during invocation."""

        class SelfRemovingListener:
            def __init__(self, transform):
                self.transform = transform
                self.calls = 0

            def on_change(self, event):
                self.calls += 1
                if self.calls == 1:
                    self.transform.remove_change_callback(self.on_change)

        t = coorx.TTransform(offset=(1, 2))
        listener = SelfRemovingListener(t)

        t.add_change_callback(listener.on_change)  # weak

        # First change - removes itself
        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 1)

        # Second change - should not be called
        t.set_params(offset=(3, 4))
        self.assertEqual(listener.calls, 1)

    def test_callback_adds_callback_during_invocation(self):
        """Callback can add new callbacks during invocation."""
        second_calls = [0]

        def second_callback(event):
            second_calls[0] += 1

        class AddingListener:
            def __init__(self, transform):
                self.transform = transform
                self.calls = 0

            def on_change(self, event):
                self.calls += 1
                if self.calls == 1:
                    self.transform.add_change_callback(second_callback, keep_reference=True)

        t = coorx.TTransform(offset=(1, 2))
        listener = AddingListener(t)

        t.add_change_callback(listener.on_change)  # weak

        t.set_params(offset=(2, 3))
        self.assertEqual(listener.calls, 1)
        self.assertEqual(second_calls[0], 0)

        # thereafter both should be called
        t.set_params(offset=(4, 5))
        self.assertEqual(listener.calls, 2)
        self.assertEqual(second_calls[0], 1)

    def test_nested_composites_all_use_weak_refs(self):
        """Nested CompositeTransforms all use weak refs and are collectible."""
        inner1 = coorx.CompositeTransform([
            coorx.TTransform(offset=(1, 0)), coorx.STTransform(scale=(2, 2), offset=(0, 0))
        ])
        inner2 = coorx.CompositeTransform([
            coorx.TTransform(offset=(0, 1)),
            coorx.AffineTransform(matrix=[[1, 0], [0, 1]], offset=(1, 1)),
        ])
        outer = coorx.CompositeTransform([inner1, inner2])

        weak_inner1 = weakref.ref(inner1)
        weak_inner2 = weakref.ref(inner2)
        weak_outer = weakref.ref(outer)

        del inner1, inner2, outer
        gc.collect()

        # All should be collected
        self.assertIsNone(weak_inner1(), "Inner1 should be collected")
        self.assertIsNone(weak_inner2(), "Inner2 should be collected")
        self.assertIsNone(weak_outer(), "Outer should be collected")

    def test_simplified_composite_uses_weak_refs(self):
        """SimplifiedCompositeTransform uses weak refs and is collectible."""
        t1 = coorx.TTransform(offset=(1, 2))
        t2 = coorx.STTransform(scale=(2, 3), offset=(4, 5))
        composite = coorx.CompositeTransform([t1, t2])

        # Access simplified version
        simplified = composite.simplified

        weak_composite = weakref.ref(composite)
        weak_simplified = weakref.ref(simplified)
        weak_t1 = weakref.ref(t1)
        weak_t2 = weakref.ref(t2)

        del t1, t2, composite, simplified
        gc.collect()

        # All should be collected
        self.assertIsNone(weak_composite(), "Composite should be collected")
        self.assertIsNone(weak_simplified(), "Simplified should be collected")
        self.assertIsNone(weak_t1(), "t1 should be collected")
        self.assertIsNone(weak_t2(), "t2 should be collected")

    def test_inverse_transform_cycle_is_collectible(self):
        """Transform and its inverse are collectible despite reference cycle."""
        t = coorx.STTransform(scale=(2, 3), offset=(1, 2))
        inv = t.inverse

        # Verify cycle exists
        self.assertIs(inv.inverse, t)

        weak_t = weakref.ref(t)
        weak_inv = weakref.ref(inv)

        del t, inv
        gc.collect()

        # Both should be collected
        self.assertIsNone(weak_t(), "Transform should be collected")
        self.assertIsNone(weak_inv(), "Inverse should be collected")

    def test_callback_invocation_order_matches_addition_order(self):
        """Callbacks should be invoked in the order they were added."""
        invocation_order = []

        def callback1(event):
            invocation_order.append(1)

        def callback2(event):
            invocation_order.append(2)

        def callback3(event):
            invocation_order.append(3)

        class MethodListener:
            def on_change(self, event):
                invocation_order.append(4)

        t = coorx.TTransform(offset=(1, 2))
        listener = MethodListener()

        # Add in specific order: function, function, method, function
        t.add_change_callback(callback1, keep_reference=True)
        t.add_change_callback(callback2, keep_reference=True)
        t.add_change_callback(listener.on_change)  # weak
        t.add_change_callback(callback3, keep_reference=True)

        # Trigger callbacks
        invocation_order.clear()
        t.set_params(offset=(2, 3))

        self.assertEqual(
            invocation_order,
            [1, 2, 4, 3],
            "Callbacks should be invoked in addition order",
        )

    def test_callback_order_preserved_after_removal(self):
        """Callback order should be preserved when removing callbacks."""
        invocation_order = []

        def callback1(event):
            invocation_order.append(1)

        def callback2(event):
            invocation_order.append(2)

        def callback3(event):
            invocation_order.append(3)

        def callback4(event):
            invocation_order.append(4)

        t = coorx.TTransform(offset=(1, 2))

        # Add four callbacks
        t.add_change_callback(callback1, keep_reference=True)
        t.add_change_callback(callback2, keep_reference=True)
        t.add_change_callback(callback3, keep_reference=True)
        t.add_change_callback(callback4, keep_reference=True)

        t.set_params(offset=(7, 8))  # Initial trigger to confirm all added
        self.assertEqual(invocation_order, [1, 2, 3, 4], "Initial callback order should be correct")

        # Remove the middle one
        t.remove_change_callback(callback2)

        # Trigger callbacks
        invocation_order.clear()
        t.set_params(offset=(2, 3))

        self.assertEqual(
            invocation_order, [1, 3, 4], "Callback order should be preserved after removal"
        )

    def test_callback_order_with_weak_refs_collected(self):
        """Callback order preserved when weak refs are collected mid-list."""
        invocation_order = []

        def callback1(event):
            invocation_order.append(1)

        def callback3(event):
            invocation_order.append(3)

        class MethodListener:
            def on_change(self, event):
                invocation_order.append(2)

        t = coorx.TTransform(offset=(1, 2))
        listener = MethodListener()

        # Add: strong, weak, strong
        t.add_change_callback(callback1, keep_reference=True)
        t.add_change_callback(listener.on_change)  # weak
        t.add_change_callback(callback3, keep_reference=True)

        # Verify initial order
        invocation_order.clear()
        t.set_params(offset=(2, 3))
        self.assertEqual(invocation_order, [1, 2, 3])

        # Delete weak listener
        del listener
        gc.collect()

        # Verify order still correct with weak ref gone
        invocation_order.clear()
        t.set_params(offset=(3, 4))
        self.assertEqual(
            invocation_order, [1, 3], "Order should be preserved when weak ref is collected"
        )


if __name__ == '__main__':
    unittest.main()
