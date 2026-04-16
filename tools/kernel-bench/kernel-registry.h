#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Element-wise kernel: works on contiguous float arrays.
// For unary ops: src1 is nullptr.
// For binary ops: both src0 and src1 are valid.
using elementwise_kernel_fn = void (*)(float * dst, const float * src0,
                                       const float * src1, int64_t n);

// MUL_MAT kernel: C[MxN] = A[MxK] * B^T[NxK]
// A is row-major [M x K], B is row-major [N x K], C is row-major [M x N]
using mul_mat_kernel_fn = void (*)(float * C, const float * A, const float * B,
                                   int64_t M, int64_t N, int64_t K);

enum class KernelOpType {
    UNARY,
    BINARY,
    MUL_MAT,
};

struct KernelEntry {
    std::string           name;       // e.g. "add", "relu", "mul_mat"
    std::string           variant;    // e.g. "standard", "custom"
    KernelOpType          op_type;
    elementwise_kernel_fn ew_fn  = nullptr;
    mul_mat_kernel_fn     mm_fn  = nullptr;
};

class KernelRegistry {
public:
    void register_kernel(const KernelEntry & entry);

    std::vector<const KernelEntry *> get_variants(const std::string & op_name) const;

    const KernelEntry * get(const std::string & op_name, const std::string & variant) const;

    std::vector<std::string> op_names() const;

    static KernelRegistry create_default();

private:
    std::unordered_map<std::string, std::unordered_map<std::string, KernelEntry>> registry_;
};
