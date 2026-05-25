#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/constants.h"

struct PrecomputedTables;
class DPMmap;  // RAII wrapper around the mmap'd cache file (defined in .cc).

// ---------------------------------------------------------------------------
// DPBuffer<T> — a vector-like buffer that can either own heap memory (used
// when DP tables are computed fresh) or refer to an externally-managed region
// (used when tables are loaded via mmap of the on-disk cache).
//
// Exposes the subset of std::vector's API that DP-table code uses:
//   data(), size(), empty(), operator[], resize(), assign(n, value).
// resize/assign always switch the buffer to owning (heap) mode.
// ---------------------------------------------------------------------------
template <typename T>
class DPBuffer {
public:
    DPBuffer() = default;
    DPBuffer(const DPBuffer&) = delete;
    DPBuffer& operator=(const DPBuffer&) = delete;

    DPBuffer(DPBuffer&& o) noexcept { steal(o); }
    DPBuffer& operator=(DPBuffer&& o) noexcept {
        if (this != &o) { release(); steal(o); }
        return *this;
    }
    ~DPBuffer() { release(); }

    // Owning, heap-backed allocation. Used by the compute path.
    void assign(std::size_t n, const T& value) {
        release();
        if (n == 0) return;
        data_  = new T[n];
        size_  = n;
        owns_  = true;
        for (std::size_t i = 0; i < n; ++i) data_[i] = value;
    }
    void resize(std::size_t n) {
        release();
        if (n == 0) return;
        data_  = new T[n]();   // value-initialise (zero for arithmetic types)
        size_  = n;
        owns_  = true;
    }

    // Non-owning view into externally-managed memory (e.g. the mmap region).
    // Lifetime of the backing memory must outlive this buffer.
    void adopt_view(const T* ptr, std::size_t n) {
        release();
        data_ = const_cast<T*>(ptr);
        size_ = n;
        owns_ = false;
    }

    T*          data()                       { return data_; }
    const T*    data()  const                { return data_; }
    std::size_t size()  const                { return size_; }
    bool        empty() const                { return size_ == 0; }
    T&          operator[](std::size_t i)       { return data_[i]; }
    const T&    operator[](std::size_t i) const { return data_[i]; }

private:
    void release() {
        if (owns_) delete[] data_;
        data_ = nullptr;
        size_ = 0;
        owns_ = false;
    }
    void steal(DPBuffer& o) noexcept {
        data_ = o.data_;
        size_ = o.size_;
        owns_ = o.owns_;
        o.data_ = nullptr;
        o.size_ = 0;
        o.owns_ = false;
    }

    T*          data_ = nullptr;
    std::size_t size_ = 0;
    bool        owns_ = false;
};

// ---------------------------------------------------------------------------
// DP Tables — six precomputed Dynamic Programming tables for Pro Yams.
//
// Tasks:
//   1. Upper section: probability of reaching cumulative target R
//   4. Upper section: expected total points (incl. progressive bonus)
//   2. Middle section: probability of no-scratch
//   5. Middle section: expected points
//   3. Lower section: probability of no-scratch
//   6. Lower section: expected points
//
// Indexed by (T, variant, section_state[, R or Sum]).
//   T:        0..78 (turns remaining)
//   variant:  one of FREE, TURBO, DOWN, UP, UPDOWN  (MID column is queried as
//             UPDOWN at runtime)
//
// Tables are persisted to disk after first computation (~2 GB on disk) and
// loaded thereafter.
// ---------------------------------------------------------------------------

constexpr int kDPNumTurns        = 79;     // T = 0..78
constexpr int kDPNumVariants     = 5;
constexpr int kDPUpperStates     = 6400;
constexpr int kDPMiddleStates    = 1690;   // 13(ss_min)×13(ls_min)×10(ss_cap)
constexpr int kDPLowerStates     = 17640;
constexpr int kDPUpperRMax       = 100;
constexpr int kDPUpperSumMax     = 105;
constexpr int kDPUpperRPad       = 104;    // R dim padded for SIMD (mult of 8)
constexpr int kDPUpperSumPad     = 112;    // Sum dim padded for SIMD

constexpr int kDPUpperCells      = 6;
constexpr int kDPMiddleCells     = 2;
constexpr int kDPLowerCells      = 5;

enum class Variant : int {
    FREE   = 0,
    TURBO  = 1,
    DOWN   = 2,
    UP     = 3,
    UPDOWN = 4,
};

struct DPVal {
    float prob_no_scratch;
    float expected_pts;
    float expected_pts_sq;
};

struct DPTables {
    // dp_t1[T][var][sc][R]   layout, R dim padded to kDPUpperRPad
    DPBuffer<float> dp_t1;
    // dp_t4[T][var][sc][S]   layout, S dim padded to kDPUpperSumPad
    DPBuffer<float> dp_t4;
    // dp_t5[T][var][sc][S]   layout, S dim padded to kDPUpperSumPad (E[X^2])
    DPBuffer<float> dp_t5;
    // dp_mid[T][var][sc]
    DPBuffer<DPVal> dp_mid;
    // dp_low[T][var][sc]
    DPBuffer<DPVal> dp_low;

    // Owns the mmap region when tables are loaded from cache. The DPBuffer
    // views above point into this region; declare it last so it outlives them
    // and on destruction is freed only after the views are released.
    std::unique_ptr<DPMmap> mapped;

    DPTables();
    ~DPTables();
    DPTables(DPTables&&) noexcept;
    DPTables& operator=(DPTables&&) noexcept;
    DPTables(const DPTables&) = delete;
    DPTables& operator=(const DPTables&) = delete;
};

// =========================================================================
// Encoders / Decoders
// Invalid (un-mapped) values in Sc clamp to index 1 (== a "0" constraint).
//
// Middle Sc convention: a 3-slot state {ss_min, ls_min, ss_cap}.
//   Sc[0] ss_min, Sc[1] ls_min each take values in
//     {-1 (filled), 0 (no constraint), 21..30 (golden threshold),
//      31 (forced scratch — mutual destruction)}.
//   Sc[2] ss_cap is the strict upper bound imposed on SS by an already-filled
//     LS (SS must be < LS): 0 (no binding cap, i.e. LS open / LS >= 30) or
//     21..29 (SS < this value). A filled LS <= 20 makes SS impossible and is
//     expressed via Sc[0]=31 instead, so the cap range starts at 21.
// At runtime, callers must pass Sc[LS]=31 if ctx.ss_scratched is true and
// LS is still empty (and likewise Sc[SS]=31 if ctx.ls_scratched is true).
// =========================================================================
int  encode_upper (const int8_t Sc[6]);
void decode_upper (int id, int8_t Sc[6]);
int  encode_middle(int8_t ss, int8_t ls, int8_t ss_cap);
void decode_middle(int id, int8_t Sc[3]);
int  encode_lower (const int8_t Sc[5]);
void decode_lower (int id, int8_t Sc[5]);

// =========================================================================
// Placement variant — return cell indices that are legal placements under v.
// N must equal kDPUpperCells / kDPMiddleCells / kDPLowerCells appropriately.
// =========================================================================
std::vector<int> get_valid_cells(Variant v, const int8_t Sc[], int N);

// =========================================================================
// Initialise tables. Loads from cache_path if it exists; else computes and
// saves. Pass empty cache_path to compute fresh without persistence.
// =========================================================================
void init_dp_tables(DPTables& dp,
                    const PrecomputedTables& tables,
                    const std::string& cache_path = "cache/dp_tables/dp_v1.bin");

// =========================================================================
// Query API — O(1) lookups after initialisation.
// =========================================================================
float get_upper_prob (const DPTables& dp, Variant v, const int8_t Sc[6], int T, int R);
float get_upper_ev   (const DPTables& dp, Variant v, const int8_t Sc[6], int T, int current_sum);
float get_upper_ev_sq(const DPTables& dp, Variant v, const int8_t Sc[6], int T, int current_sum);
float get_middle_prob(const DPTables& dp, Variant v, const int8_t Sc[3], int T);
float get_middle_ev  (const DPTables& dp, Variant v, const int8_t Sc[3], int T);
float get_middle_ev_sq(const DPTables& dp, Variant v, const int8_t Sc[3], int T);
float get_lower_prob (const DPTables& dp, Variant v, const int8_t Sc[5], int T);
float get_lower_ev   (const DPTables& dp, Variant v, const int8_t Sc[5], int T);
float get_lower_ev_sq(const DPTables& dp, Variant v, const int8_t Sc[5], int T);

// =========================================================================
// Helpers exposed for tests.
// =========================================================================
inline int upper_bonus(int sum) {
    return sum >= 100 ? 500
         : sum >=  90 ? 200
         : sum >=  80 ? 100
         : sum >=  70 ?  50
         : sum >=  60 ?  30
         : 0;
}
