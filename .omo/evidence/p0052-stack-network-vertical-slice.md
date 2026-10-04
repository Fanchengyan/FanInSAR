# Proposal 0052 Stack Network vertical slice

## Scenario

Build the synthetic one-pair Stack fixture in `tests/stack/test_spatial_unwrap_api.py`,
commit its unwrapped generation with `_IdentityUnwrapper`, obtain `stack.network`,
and invert the live Network view directly. The scenario does not call
`publish_generation()` or `Network.open()`.

## Red/green evidence

The existing Stack surface already exposed the live view after the complete unwrap
generation. The missing regression was the direct `stack.network.invert()` call;
the new test is:

```text
tests/stack/test_spatial_unwrap_api.py::test_stack_network_inverts_without_publish_or_reopen
```

Invocation:

```text
.venv/bin/pytest tests/stack/test_spatial_unwrap_api.py::test_stack_network_inverts_without_publish_or_reopen -q
```

Binary observable: `1 passed`, with `result.pair_ids == ("20240101_20240113",)`
and `result.phase_cumulative_rad.shape == (2, 2, 3)`.

Focused regression invocation:

```text
.venv/bin/pytest tests/stack/test_spatial_unwrap_api.py tests/stack/test_session.py::test_live_stack_network_view_exposes_committed_ifg_pairs tests/stack/test_session.py::test_publish_generation_exports_canonical_network_view -q
```

Binary observable: `10 passed`.

Static validation invocation:

```text
.venv/bin/ruff check faninsar/stack/session.py
.venv/bin/ruff format --check faninsar/stack/session.py tests/stack/test_spatial_unwrap_api.py
```

Binary observable: Ruff check passed and both files were already formatted.
