#include "solver/dp_tables.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "engine/constants.h"
#include "solver/dp_tables_internal.h"
#include "solver/precomputed_tables.h"

// ===========================================================================
// DPMmap — RAII wrapper around an mmap'd cache file.
// The DPBuffer views inside DPTables point into base().
// ===========================================================================
class DPMmap {
public:
    DPMmap(int fd, void* base, std::size_t len) : fd_(fd), base_(base), len_(len) {}
    DPMmap(const DPMmap&) = delete;
    DPMmap& operator=(const DPMmap&) = delete;
    ~DPMmap() {
        if (base_ && base_ != MAP_FAILED) ::munmap(base_, len_);
        if (fd_ >= 0) ::close(fd_);
    }
    void*       base() const { return base_; }
    std::size_t len()  const { return len_; }

private:
    int         fd_   = -1;
    void*       base_ = nullptr;
    std::size_t len_  = 0;
};

// Out-of-line so the destructor sees the complete DPMmap type.
DPTables::DPTables() = default;
DPTables::~DPTables() = default;
DPTables::DPTables(DPTables&&) noexcept = default;
DPTables& DPTables::operator=(DPTables&&) noexcept = default;

// ===========================================================================
// Forward declarations from dp_tables_compute.cc
// ===========================================================================
void compute_upper_dp (DPTables& dp, const PrecomputedTables& tables);
void compute_middle_dp(DPTables& dp, const PrecomputedTables& tables);
void compute_lower_dp (DPTables& dp, const PrecomputedTables& tables);

// ===========================================================================
// Encoders / Decoders
// ===========================================================================

namespace {

inline int enc_1s(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case  3: return 2;
        case  4: return 3;
        case  5: return 4;
        default: return 1;
    }
}
inline int enc_2s(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case  6: return 2;
        case  8: return 3;
        case 10: return 4;
        default: return 1;
    }
}
inline int enc_3s(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 12: return 2;
        case 15: return 3;
        default: return 1;
    }
}
inline int enc_4s(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 16: return 2;
        case 20: return 3;
        default: return 1;
    }
}
inline int enc_5s(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 20: return 2;
        case 25: return 3;
        default: return 1;
    }
}
inline int enc_6s(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 24: return 2;
        case 30: return 3;
        default: return 1;
    }
}

constexpr int8_t kVals1s[] = {-1, 0, 3, 4, 5};
constexpr int8_t kVals2s[] = {-1, 0, 6, 8, 10};
constexpr int8_t kVals3s[] = {-1, 0, 12, 15};
constexpr int8_t kVals4s[] = {-1, 0, 16, 20};
constexpr int8_t kVals5s[] = {-1, 0, 20, 25};
constexpr int8_t kVals6s[] = {-1, 0, 24, 30};

inline int enc_mid(int8_t v) {
    if (v == -1) return 0;
    if (v == 0)  return 1;
    if (v >= 21 && v <= 30) return v - 19;  // 21->2 .. 30->11
    if (v == 31) return 12;                 // forced scratch sentinel
    return 1;
}
constexpr int8_t kValsMid[] = {-1, 0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

inline int enc_fh(int8_t v) {
    if (v == -1) return 0;
    if (v == 0)  return 1;
    if (v == 27) return 2;
    if (v == 28) return 3;
    if (v >= 30 && v <= 48) return 4 + (v - 30);  // 30->4 .. 48->22
    if (v == 50) return 23;
    return 1;
}
inline int enc_k(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 38: return 2;
        case 42: return 3;
        case 46: return 4;
        case 50: return 5;
        case 54: return 6;
        default: return 1;
    }
}
inline int enc_q(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 50: return 2;
        default: return 1;  // 45 (and others) drop to "no constraint"
    }
}
inline int enc_u8(int8_t v) {
    switch (v) {
        case -1: return 0;
        case  0: return 1;
        case 65: return 2;
        case 70: return 3;
        case 75: return 4;
        default: return 1;
    }
}
inline int enc_y(int8_t v) {
    switch (v) {
        case -1: return 0;
        case   0: return 1;
        case  80: return 2;
        case  85: return 3;
        case  90: return 4;
        case  95: return 5;
        case 100: return 6;
        default: return 1;
    }
}

constexpr int8_t kValsFH[] = {-1, 0, 27, 28,
                              30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                              40, 41, 42, 43, 44, 45, 46, 47, 48,
                              50};
constexpr int8_t kValsK[]  = {-1, 0, 38, 42, 46, 50, 54};
constexpr int8_t kValsQ[]  = {-1, 0, 50};
constexpr int8_t kValsU8[] = {-1, 0, 65, 70, 75};
constexpr int8_t kValsY[]  = {-1, 0, 80, 85, 90, 95, 100};

}  // namespace

int encode_upper(const int8_t Sc[6]) {
    return enc_1s(Sc[0])
         + 5 * (enc_2s(Sc[1])
              + 5 * (enc_3s(Sc[2])
                   + 4 * (enc_4s(Sc[3])
                        + 4 * (enc_5s(Sc[4])
                             + 4 * enc_6s(Sc[5])))));
}

void decode_upper(int id, int8_t Sc[6]) {
    Sc[0] = kVals1s[id % 5]; id /= 5;
    Sc[1] = kVals2s[id % 5]; id /= 5;
    Sc[2] = kVals3s[id % 4]; id /= 4;
    Sc[3] = kVals4s[id % 4]; id /= 4;
    Sc[4] = kVals5s[id % 4]; id /= 4;
    Sc[5] = kVals6s[id % 4];
}

int encode_middle(int8_t ss, int8_t ls) {
    return enc_mid(ss) + 13 * enc_mid(ls);
}

void decode_middle(int id, int8_t Sc[2]) {
    Sc[0] = kValsMid[id % 13];
    Sc[1] = kValsMid[id / 13];
}

int encode_lower(const int8_t Sc[5]) {
    return enc_fh(Sc[0])
         + 24 * (enc_k(Sc[1])
              + 7 * (enc_q(Sc[2])
                   + 3 * (enc_u8(Sc[3])
                        + 5 * enc_y(Sc[4]))));
}

void decode_lower(int id, int8_t Sc[5]) {
    Sc[0] = kValsFH[id % 24]; id /= 24;
    Sc[1] = kValsK [id %  7]; id /=  7;
    Sc[2] = kValsQ [id %  3]; id /=  3;
    Sc[3] = kValsU8[id %  5]; id /=  5;
    Sc[4] = kValsY [id %  7];
}

// ===========================================================================
// Variant logic
// ===========================================================================
std::vector<int> get_valid_cells(Variant v, const int8_t Sc[], int N) {
    std::vector<int> cells;
    cells.reserve(N);

    switch (v) {
        case Variant::FREE:
        case Variant::TURBO:
            for (int i = 0; i < N; ++i)
                if (Sc[i] != -1) cells.push_back(i);
            return cells;

        case Variant::DOWN:
            for (int i = 0; i < N; ++i)
                if (Sc[i] != -1) { cells.push_back(i); break; }
            return cells;

        case Variant::UP:
            for (int i = N - 1; i >= 0; --i)
                if (Sc[i] != -1) { cells.push_back(i); break; }
            return cells;

        case Variant::UPDOWN: {
            bool any_filled = false;
            for (int i = 0; i < N; ++i) {
                if (Sc[i] == -1) { any_filled = true; break; }
            }
            if (!any_filled) {
                for (int i = 0; i < N; ++i) cells.push_back(i);
                return cells;
            }
            for (int i = 0; i < N; ++i) {
                if (Sc[i] == -1) continue;
                bool adj_left  = (i > 0)       && (Sc[i - 1] == -1);
                bool adj_right = (i + 1 < N)   && (Sc[i + 1] == -1);
                if (adj_left || adj_right) cells.push_back(i);
            }
            return cells;
        }
    }
    return cells;
}

// ===========================================================================
// Query API
// ===========================================================================
static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

float get_upper_prob(const DPTables& dp, Variant v, const int8_t Sc[6], int T, int R) {
    if (R <= 0) return 1.0f;  // target already met
    int sc = encode_upper(Sc);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    int r  = clamp_int(R, 0, kDPUpperRMax);
    return dp.dp_t1[dp_idx_t1(t, static_cast<int>(v), sc, r)];
}

float get_upper_ev(const DPTables& dp, Variant v, const int8_t Sc[6], int T, int current_sum) {
    int sc = encode_upper(Sc);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    int s  = clamp_int(current_sum, 0, kDPUpperSumMax);
    return dp.dp_t4[dp_idx_t4(t, static_cast<int>(v), sc, s)];
}

float get_middle_prob(const DPTables& dp, Variant v, const int8_t Sc[2], int T) {
    int sc = encode_middle(Sc[0], Sc[1]);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    return dp.dp_mid[dp_idx_mid(t, static_cast<int>(v), sc)].prob_no_scratch;
}

float get_middle_ev(const DPTables& dp, Variant v, const int8_t Sc[2], int T) {
    int sc = encode_middle(Sc[0], Sc[1]);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    return dp.dp_mid[dp_idx_mid(t, static_cast<int>(v), sc)].expected_pts;
}

float get_lower_prob(const DPTables& dp, Variant v, const int8_t Sc[5], int T) {
    int sc = encode_lower(Sc);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    return dp.dp_low[dp_idx_low(t, static_cast<int>(v), sc)].prob_no_scratch;
}

float get_lower_ev(const DPTables& dp, Variant v, const int8_t Sc[5], int T) {
    int sc = encode_lower(Sc);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    return dp.dp_low[dp_idx_low(t, static_cast<int>(v), sc)].expected_pts;
}

float get_upper_ev_sq(const DPTables& dp, Variant v, const int8_t Sc[6], int T, int current_sum) {
    int sc = encode_upper(Sc);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    int s  = clamp_int(current_sum, 0, kDPUpperSumMax);
    return dp.dp_t5[dp_idx_t4(t, static_cast<int>(v), sc, s)];
}

float get_middle_ev_sq(const DPTables& dp, Variant v, const int8_t Sc[2], int T) {
    int sc = encode_middle(Sc[0], Sc[1]);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    return dp.dp_mid[dp_idx_mid(t, static_cast<int>(v), sc)].expected_pts_sq;
}

float get_lower_ev_sq(const DPTables& dp, Variant v, const int8_t Sc[5], int T) {
    int sc = encode_lower(Sc);
    int t  = clamp_int(T, 0, kDPNumTurns - 1);
    return dp.dp_low[dp_idx_low(t, static_cast<int>(v), sc)].expected_pts_sq;
}

// ===========================================================================
// Disk persistence
// ===========================================================================
namespace {

constexpr uint32_t kCacheMagic   = 0x59414D44;  // "DMAY"
constexpr uint32_t kCacheVersion = 4;           // v4: SS forced-scratch under filled LS (middle DP)

struct CacheHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t dp_t1_count;   // floats
    uint64_t dp_t4_count;
    uint64_t dp_t5_count;
    uint64_t dp_mid_count;  // DPVal entries
    uint64_t dp_low_count;
};

// Map the cache file read-only and point the DPBuffer views at it. Pages are
// shared via the kernel page cache across processes, and unmapped on
// DPTables destruction. Demand-paged: first access to a page triggers a
// minor fault to bring it in from the page cache (or a major fault on first
// run after boot).
bool load_from_cache(DPTables& dp, const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;

    struct stat st;
    if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
    const std::size_t flen = static_cast<std::size_t>(st.st_size);
    if (flen < sizeof(CacheHeader)) { ::close(fd); return false; }

    void* base = ::mmap(nullptr, flen, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (base == MAP_FAILED) { ::close(fd); return false; }
    // DP lookups are scattered; disable kernel read-ahead.
    ::madvise(base, flen, MADV_RANDOM);

    auto fail = [&]() {
        ::munmap(base, flen);
        ::close(fd);
        return false;
    };

    CacheHeader h{};
    std::memcpy(&h, base, sizeof(h));
    if (h.magic != kCacheMagic || h.version != kCacheVersion) return fail();

    const uint64_t exp_t1  = static_cast<uint64_t>(kDPNumTurns) * kDPNumVariants
                             * kDPUpperStates * kDPUpperRPad;
    const uint64_t exp_t4  = static_cast<uint64_t>(kDPNumTurns) * kDPNumVariants
                             * kDPUpperStates * kDPUpperSumPad;
    const uint64_t exp_mid = static_cast<uint64_t>(kDPNumTurns) * kDPNumVariants
                             * kDPMiddleStates;
    const uint64_t exp_low = static_cast<uint64_t>(kDPNumTurns) * kDPNumVariants
                             * kDPLowerStates;
    if (h.dp_t1_count != exp_t1 || h.dp_t4_count != exp_t4
        || h.dp_t5_count != exp_t4 || h.dp_mid_count != exp_mid
        || h.dp_low_count != exp_low) {
        return fail();
    }

    const std::size_t expected_size = sizeof(CacheHeader)
                                    + exp_t1 * sizeof(float)
                                    + exp_t4 * sizeof(float) * 2
                                    + exp_mid * sizeof(DPVal)
                                    + exp_low * sizeof(DPVal);
    if (flen < expected_size) return fail();

    auto* p = static_cast<const char*>(base) + sizeof(CacheHeader);
    dp.dp_t1 .adopt_view(reinterpret_cast<const float*>(p), exp_t1);
    p += exp_t1 * sizeof(float);
    dp.dp_t4 .adopt_view(reinterpret_cast<const float*>(p), exp_t4);
    p += exp_t4 * sizeof(float);
    dp.dp_t5 .adopt_view(reinterpret_cast<const float*>(p), exp_t4);
    p += exp_t4 * sizeof(float);
    dp.dp_mid.adopt_view(reinterpret_cast<const DPVal*>(p), exp_mid);
    p += exp_mid * sizeof(DPVal);
    dp.dp_low.adopt_view(reinterpret_cast<const DPVal*>(p), exp_low);

    dp.mapped = std::make_unique<DPMmap>(fd, base, flen);
    return true;
}

bool save_to_cache(const DPTables& dp, const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) return false;
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;

    CacheHeader h{
        kCacheMagic, kCacheVersion,
        dp.dp_t1.size(), dp.dp_t4.size(), dp.dp_t5.size(),
        dp.dp_mid.size(), dp.dp_low.size()
    };
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(dp.dp_t1.data()),
            dp.dp_t1.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(dp.dp_t4.data()),
            dp.dp_t4.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(dp.dp_t5.data()),
            dp.dp_t5.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(dp.dp_mid.data()),
            dp.dp_mid.size() * sizeof(DPVal));
    f.write(reinterpret_cast<const char*>(dp.dp_low.data()),
            dp.dp_low.size() * sizeof(DPVal));
    return static_cast<bool>(f);
}

}  // namespace

// ===========================================================================
// init_dp_tables — top-level coordinator
// ===========================================================================
void init_dp_tables(DPTables& dp,
                    const PrecomputedTables& tables,
                    const std::string& cache_path) {
    if (!cache_path.empty() && load_from_cache(dp, cache_path)) {
        return;
    }

    // Allocate.
    dp.dp_t1.assign(static_cast<std::size_t>(kDPNumTurns) * kDPNumVariants
                        * kDPUpperStates * kDPUpperRPad, 0.0f);
    dp.dp_t4.assign(static_cast<std::size_t>(kDPNumTurns) * kDPNumVariants
                        * kDPUpperStates * kDPUpperSumPad, 0.0f);
    dp.dp_t5.assign(static_cast<std::size_t>(kDPNumTurns) * kDPNumVariants
                        * kDPUpperStates * kDPUpperSumPad, 0.0f);
    dp.dp_mid.assign(static_cast<std::size_t>(kDPNumTurns) * kDPNumVariants
                        * kDPMiddleStates, DPVal{0.0f, 0.0f, 0.0f});
    dp.dp_low.assign(static_cast<std::size_t>(kDPNumTurns) * kDPNumVariants
                        * kDPLowerStates, DPVal{0.0f, 0.0f, 0.0f});

    compute_upper_dp(dp, tables);
    compute_middle_dp(dp, tables);
    compute_lower_dp(dp, tables);

    if (!cache_path.empty()) {
        save_to_cache(dp, cache_path);
    }
}
