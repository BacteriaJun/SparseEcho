# Design notes

This file records the design path that led to the current 1.1 architecture. It is intentionally about technical decisions rather than release history.

## Residual-support recovery was not enough

An early direction treated the problem mainly as dynamic sparse support recovery: explain persistent support, search residuals for new support, then track. Residual cancellation remains useful engineering, but existing LS-CS-residual/KF-CS-style work makes it a poor defining principle for SparseEcho.

## High-order invariants were too expensive

Higher-order invariant/cumulant constructions can suppress nuisance parameters algebraically, but the sample-complexity and variance cost was unattractive for the intended weak-signal boundary. They were not carried into the public runtime.

## Tensor factorization required too much endpoint structure

A tensor/partial-trace route could remove nuisance factors cleanly, but it assumed multiple identity views at the endpoint. That assumption was stronger than the hardware-neutral acquisition contract should require.

## The surviving question

The useful design question became:

> Can the physical execution schedule be chosen so unavoidable time variation has a compact representation in the same transform domain used for sparse support recovery?

That is the basis for SFPTI.

## Why Gray is the default

Natural binary and Gray execution both reach the `r`-generator time-support lower bound for an `r`-bit local aperture. Gray remains the default because it also has one-bit adjacent transitions. The implementation keeps binary and random ordering as controls.

## Local proposals are intentionally bounded

The local stage is a proposal generator, not the final identity detector. Version 1.1 uses a CFAR threshold and a separate compute budget. Proposal-budget saturation is treated as an operating signal rather than disguised as a statistical decision.

## Receiver-space evidence is an independent observable

Hash incidence alone produces too many combinatorial global candidates at the current public view count. Multi-receiver structure provides an additional observable. The runtime supports a configurable low-rank subspace model rather than assuming exact rank-1 persistence.

Large cross-view rotation is still a real failure region and is kept visible in the stored validation data.

## Why the automatic controller shortens the aperture

Multi-pass virtual-time synthesis is mathematically useful when the temporal model and timing geometry support it. For rapidly rotating complex signals, however, collecting repeated passes over a longer physical interval can reduce coherence. The default runtime therefore closes the tail budget by shortening the aperture. VTQC remains an explicit primitive for controlled integrations.
