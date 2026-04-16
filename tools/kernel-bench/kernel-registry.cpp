#include "kernel-registry.h"
#include "kernels/kernels-standard.h"
#include "kernels/kernels-custom.h"

void KernelRegistry::register_kernel(const KernelEntry & entry) {
    registry_[entry.name][entry.variant] = entry;
}

std::vector<const KernelEntry *> KernelRegistry::get_variants(const std::string & op_name) const {
    std::vector<const KernelEntry *> result;
    auto it = registry_.find(op_name);
    if (it != registry_.end()) {
        for (auto & [variant_name, entry] : it->second) {
            result.push_back(&entry);
        }
    }
    return result;
}

const KernelEntry * KernelRegistry::get(const std::string & op_name, const std::string & variant) const {
    auto it = registry_.find(op_name);
    if (it == registry_.end()) return nullptr;
    auto vit = it->second.find(variant);
    if (vit == it->second.end()) return nullptr;
    return &vit->second;
}

std::vector<std::string> KernelRegistry::op_names() const {
    std::vector<std::string> names;
    for (auto & [name, _] : registry_) {
        names.push_back(name);
    }
    // Stable ordering for consistent output
    std::sort(names.begin(), names.end());
    return names;
}

KernelRegistry KernelRegistry::create_default() {
    KernelRegistry reg;

    // Standard (baseline) kernels
    reg.register_kernel({"add",     "standard", KernelOpType::BINARY,  kernel_add_standard,     nullptr});
    reg.register_kernel({"mul",     "standard", KernelOpType::BINARY,  kernel_mul_standard,     nullptr});
    reg.register_kernel({"relu",    "standard", KernelOpType::UNARY,   kernel_relu_standard,    nullptr});
    reg.register_kernel({"sigmoid", "standard", KernelOpType::UNARY,   kernel_sigmoid_standard, nullptr});
    reg.register_kernel({"silu",    "standard", KernelOpType::UNARY,   kernel_silu_standard,    nullptr});
    reg.register_kernel({"mul_mat", "standard", KernelOpType::MUL_MAT, nullptr,                 kernel_mul_mat_standard});

    // Custom kernels (replace implementations in kernels/kernels-custom.cpp)
    reg.register_kernel({"add",     "custom",   KernelOpType::BINARY,  kernel_add_custom,       nullptr});
    reg.register_kernel({"mul",     "custom",   KernelOpType::BINARY,  kernel_mul_custom,       nullptr});
    reg.register_kernel({"relu",    "custom",   KernelOpType::UNARY,   kernel_relu_custom,      nullptr});
    reg.register_kernel({"sigmoid", "custom",   KernelOpType::UNARY,   kernel_sigmoid_custom,   nullptr});
    reg.register_kernel({"silu",    "custom",   KernelOpType::UNARY,   kernel_silu_custom,      nullptr});
    reg.register_kernel({"mul_mat", "custom",   KernelOpType::MUL_MAT, nullptr,                 kernel_mul_mat_custom});

    return reg;
}
