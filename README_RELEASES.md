# Release model

SparseEcho is an open-source extraction of a reconstruction component used inside a larger engineering program. Public scope may expand release by release as integration and disclosure boundaries permit.

Versioning is intentionally simple:

- **X.0** — first public engineering prototype for generation X. The core method is usable, but the deployment loop may still rely on external integration.
- **X.1** — operational release for generation X. Acquisition, reconstruction, calibration, fault handling and control feedback are closed at the public software boundary.
- **X.N** — incremental releases within the same generation. They refine implementation, validation and integration contracts without changing the generation's core method.

The repository documents only the public reconstruction boundary. System-specific hardware and deployment layers remain outside the release unless separately authorized.
