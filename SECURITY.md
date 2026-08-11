# Security

SparseEcho processes externally supplied capture data and query-plan metadata. Treat capture directories and runtime messages as untrusted input.

For deployment integrations:

- verify the query-plan fingerprint before accepting a frame;
- bound receiver count, slot count and metadata sizes at the transport boundary;
- keep raw captures outside privileged control processes;
- do not load executable calibration data or Python objects from capture metadata;
- isolate acquisition drivers from the reconstruction worker where practical;
- persist runtime fault and calibration epochs for post-run traceability.

Please report security issues privately to the repository maintainer rather than opening a public issue with deployment-specific details.
