# Network generation binding evidence

## Scope

Persisted Stack Network generation IDs now hash tagged IFG and unwrap manifest
digests. Analysis recomputes the identity from current stores and verifies each
current unwrap digest and `(IFG digest, unwrap digest)` lineage against the
admitted `UNWRAPPED_PHASE` Network product. Replacing one unwrap `CURRENT`
therefore fails closed before the solver runs.

## Verification

- Invocation: `uv run ruff check faninsar/processing/stack/session.py tests/processing/stack/test_stack_session.py`
  - Observable: `All checks passed!`
- Invocation: `uv run pytest tests/processing/stack/test_stack_session.py tests/core/test_network_contract.py -q --override-ini='addopts='`
  - Observable: `52 passed, 6 warnings`
- Invocation: `uv run pytest tests/datasets/network/test_network.py tests/processing/stack/test_stack_session.py tests/core/test_network_contract.py -q --override-ini='addopts='`
  - Observable: `70 passed, 6 warnings`
- Focused regression: `test_stack_analysis_rejects_replaced_unwrap_generation`
  publishes a replacement unwrap generation for one pair, then invokes
  `stack.analyze_time_series()` and observes `InvalidProcessingStateError`
  before solver admission because the Network product digest/lineage no longer
  matches current unwrap state.
