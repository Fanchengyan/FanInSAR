# Network unwrapped-product publication evidence

## Scope

`Stack._refresh_network_from_ifg_dirs` now treats unwrap artifacts as one
all-or-nothing set. After a successful Stack unwrap, the Network product index
contains both the manifest-bound complex interferogram and unwrapped-phase
record for every configured pair. A partial set is rejected before the Network
generation is replaced.

## Verification

- Invocation: `uv run ruff check faninsar/processing/stack/session.py tests/processing/stack/test_stack_session.py`
  - Observable: `All checks passed!`
- Invocation: `uv run pytest tests/processing/stack/test_stack_session.py -q --override-ini='addopts='`
  - Observable: `35 passed, 5 warnings`
  - Focused complete-set assertion: `test_stack_unwrap_and_sbas_load_persisted_pair_artifacts`
    observes 3 `COMPLEX_INTERFEROGRAM` and 3 `UNWRAPPED_PHASE` products, with
    non-empty content digests and lineage; each unwrapped record carries the
    IFG and unwrap manifest digests in its lineage.
  - Focused partial-set assertion: `test_partial_unwrap_network_cannot_publish_stack_generation`
    invokes `_refresh_network_from_ifg_dirs()` with one unwrap manifest across
    three IFGs and observes `InvalidProcessingStateError` matching
    `partial unwrap product set`; `stack.analysis_ready` remains false.
- Invocation: `uv run pytest tests/datasets/network/test_network.py tests/processing/stack/test_stack_session.py -q --override-ini='addopts='`
  - Observable: `53 passed, 5 warnings`
