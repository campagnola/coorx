<!-- Maintained by the `architecture-friction` agent skill. Humans welcome. -->

# Architecture friction

Places where this codebase's structure made a change harder than it should have
been, and the refactors that would relieve them.

This is **not** a bug tracker or a TODO list. Every entry answers one question:
*what would have made a change we actually made easy?*

- `## <a refactor direction>` — states a direction ("rework device locking so it
  can span threads"), not a specific design and not a place. Under it, one dated
  bullet per time that friction was felt.
- An occurrence may sit under several headings if any of those refactors would
  have relieved it. Its `(slug)` is identical in each, so all copies can be
  found and removed together when one lands.
- `## Unfiled` — raw notes captured mid-work, not yet sorted. An unsorted note
  still beats a forgotten one.
- `## Done` — refactors that landed, with dates, so the same friction doesn't
  get re-logged from scratch.

**Three bullets under one heading means the refactor is due.**

Adding an entry costs one line. Please add them.

## Let CompositeTransform map objects through the base Transform path

- [2026-01-07] (composite-dims-reversed) `CompositeTransform.dims` reports its
  endpoints in the wrong order, which is what keeps the chain from using
  `_prepare_and_map`: routing through it would trip the base class's dimension
  check on any chain that changes dimensionality.
- [2026-10-26] (systemless-products-point-check) Making products system-less
  meant re-implementing the base class's `_coorx_transform` dispatch inside
  `CompositeTransform.map`, because the chain maps objects itself instead of
  going through `_prepare_and_map` like every other transform.

## Unfiled
