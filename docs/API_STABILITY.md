# API and ABI stability

SparseEcho separates deployment contracts from reference implementation internals.

## Stable for the 1.x generation

The following interfaces are compatibility surfaces:

- capture manifest fields and slot-major `complex64` transport semantics;
- query-plan fingerprint construction and challenge ordering metadata;
- runtime fault codes and accept/reacquire/reject decision semantics;
- calibration epoch, causal switch-response taps and receiver-gain snapshot fields;
- C ABI major/minor rules in `include/sparseecho/runtime.h`;
- published protobuf field numbers in `proto/sparseecho_runtime.proto`;
- JSON schema field meanings under `schemas/`.

Compatible 1.x changes may add optional fields. They must not silently change the meaning of an existing field or reuse a protobuf field number.

## Reference API

The Python package is the executable reference and follows normal semantic-version compatibility within 1.x, but individual recovery classes, proposal data structures and validation helpers are not a deployment ABI. Integrations should enter through `ReconstructionRuntime`, the capture/calibration contracts, or an equivalent non-Python binding.

## Experimental surface

The independent forward generator, benchmark scenario definitions and low-level recovery diagnostics are validation tools. Their configuration may evolve between minor releases when that improves model coverage or exposes a previously hidden assumption.

## ABI negotiation

C records carry `struct_size`, `abi_major` and `abi_minor`. Consumers should reject an unknown major version and may accept a newer minor version when the known structure prefix is large enough for the fields they consume.
