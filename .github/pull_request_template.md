## Proposal Compliance

- Applicable Proposal IDs: `PROPOSAL-####` values from applicable registered Proposals, or `none`
- [ ] I identified the applicable Proposal contract(s), or confirmed that none apply.
- [ ] I read and comply with each applicable Proposal contract; any intentional divergence includes the required Proposal revision or new Proposal.

### Conditional geometry/native checks

Complete this section only when the PR changes geometry, native code, packaging, or related CI:

- [ ] Applicable geometry/native Proposal IDs are listed above.
- [ ] Native inputs and results use centralized validation and preserve the applicable fourteen-field result contract.
- [ ] Native/Compile/Eager comparisons use the same fixture, variable DEM, iteration budget, and I/O boundary.
- [ ] Any Native/OpenMP qualification evidence comes from the actual geometry loop; serial CI smoke is not a performance claim.
- [ ] No retired finite-difference Newton path or unqualified Native performance claim was introduced.

## Verification

<!-- List the exact verification commands and concise results. Include relevant geometry/native workflow commands when applicable. -->
