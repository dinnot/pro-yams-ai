#pragma once

// Shared PrecomputedTables + libtorch thread-clamp helper for the distil
// test suite. Both `at::set_num_interop_threads` and the tables init can
// only happen ONCE per process — the static-local + std::once_flag pattern
// lets each test file call `distil_test::tables()` freely without worrying
// about who beat them to the init.

#include <mutex>

#include <torch/torch.h>

#include "solver/precomputed_tables.h"

namespace distil_test {

inline PrecomputedTables& tables() {
    static PrecomputedTables* g = []() {
        // Clamping must come before any parallel work — including the
        // single-shot calls another test file may have already made.
        torch::set_num_threads(1);
        try {
            at::set_num_interop_threads(1);
        } catch (...) {
            // Already set (by an earlier test executable in the same process,
            // or a test file linked into the same binary). Ignore.
        }
        auto* p = new PrecomputedTables();
        init_precomputed_tables(*p);
        return p;
    }();
    return *g;
}

}  // namespace distil_test
