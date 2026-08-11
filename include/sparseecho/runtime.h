#ifndef SPARSEECHO_RUNTIME_H
#define SPARSEECHO_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARSEECHO_ABI_MAJOR 1u
#define SPARSEECHO_ABI_MINOR 1u

typedef struct se_complex32 {
    float real;
    float imag;
} se_complex32;

typedef struct se_capture_frame {
    uint32_t struct_size;
    uint16_t abi_major;
    uint16_t abi_minor;
    const se_complex32 *slots;      /* slot-major, receiver-minor */
    const uint8_t *valid;           /* one byte per physical slot */
    const int64_t *slot_time_ns;    /* optional; NULL if unavailable */
    uint32_t physical_slots;
    uint16_t receivers;
    uint16_t pass_count;
    uint64_t sequence_id;
    uint64_t monotonic_start_ns;
    uint64_t monotonic_end_ns;
    double aperture_scale;
    char plan_fingerprint[65];      /* lowercase SHA-256 hex + NUL */
    char calibration_epoch[64];
} se_capture_frame;

typedef struct se_identity_result {
    uint32_t identity;
    uint16_t view_support;
    uint16_t reserved;
    float spatial_consistency;
} se_identity_result;

typedef struct se_reconstruction_result {
    uint32_t struct_size;
    uint16_t abi_major;
    uint16_t abi_minor;
    const se_identity_result *items;
    uint32_t count;
    uint32_t candidate_count;
    uint32_t pass_count;
    uint32_t flags;
} se_reconstruction_result;

typedef enum se_runtime_action {
    SE_RUNTIME_ACCEPT = 0,
    SE_RUNTIME_REACQUIRE = 1,
    SE_RUNTIME_REJECT = 2
} se_runtime_action;

typedef enum se_fault_code {
    SE_FAULT_NONE = 0,
    SE_FAULT_PLAN_MISMATCH = 1,
    SE_FAULT_SEQUENCE_REGRESSION = 2,
    SE_FAULT_SLOT_COUNT_MISMATCH = 3,
    SE_FAULT_NONFINITE_INPUT = 4,
    SE_FAULT_INVALID_TIMING = 5,
    SE_FAULT_TIMING_JITTER = 6,
    SE_FAULT_TOO_MANY_ERASURES = 7,
    SE_FAULT_ERASURE_BURST = 8,
    SE_FAULT_RECEIVER_COUNT = 9,
    SE_FAULT_CALIBRATION = 10,
    SE_FAULT_ACQUISITION = 11,
    SE_FAULT_RECONSTRUCTION = 12,
    SE_FAULT_REACQUIRE_EXHAUSTED = 13
} se_fault_code;

typedef struct se_runtime_fault {
    uint32_t struct_size;
    uint16_t abi_major;
    uint16_t abi_minor;
    se_fault_code code;
    uint8_t recoverable;
    uint8_t reserved[3];
    char detail[256];
} se_runtime_fault;

typedef struct se_runtime_decision {
    uint32_t struct_size;
    uint16_t abi_major;
    uint16_t abi_minor;
    se_runtime_action action;
    double aperture_scale_factor;
    double measured_phase_span_cycles;
    double allowed_phase_span_cycles;
    double fiber_tail_energy;
} se_runtime_decision;

/*
 * struct_size and ABI fields permit compatible extensions without relying on
 * compiler-specific structure discovery. This header defines the public data
 * boundary only; buffer transport and ownership remain deployment concerns.
 */

#ifdef __cplusplus
}
#endif

#endif /* SPARSEECHO_RUNTIME_H */
