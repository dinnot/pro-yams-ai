#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "solver/dp_tables.h"
#include "solver/precomputed_tables.h"

// ===========================================================================
// Heavy DP monotonicity / sanity test.
//
// Computes (or loads) the full DP tables and verifies coarse correctness
// properties: probabilities in [0,1], EVs non-negative, monotonicity in T
// from the empty board.
//
// Cache path can be overridden via env var DP_TABLES_CACHE.
// ===========================================================================

namespace {

DPTables& shared_dp() {
    static PrecomputedTables pt;
    static DPTables dp;
    static bool initialized = false;
    if (!initialized) {
        init_precomputed_tables(pt);
        const char* env = std::getenv("DP_TABLES_CACHE");
        std::string path = env ? env : "cache/dp_tables/dp_v1.bin";
        init_dp_tables(dp, pt, path);
        initialized = true;
    }
    return dp;
}

constexpr int kVariants[] = {
    static_cast<int>(Variant::FREE),
    static_cast<int>(Variant::TURBO),
    static_cast<int>(Variant::DOWN),
    static_cast<int>(Variant::UP),
    static_cast<int>(Variant::UPDOWN),
};

}  // namespace

TEST(DPMonotonicityTest, UpperProbInRangeFromEmpty) {
    // T must be >= 6 (need at least one turn per unfilled cell to be feasible).
    const DPTables& dp = shared_dp();
    int8_t Sc[6] = {0, 0, 0, 0, 0, 0};
    for (int v : kVariants) {
        for (int T = 6; T <= 78; ++T) {
            for (int R = 0; R <= 100; R += 10) {
                float p = get_upper_prob(dp, static_cast<Variant>(v), Sc, T, R);
                EXPECT_GE(p, 0.0f) << "v=" << v << " T=" << T << " R=" << R;
                EXPECT_LE(p, 1.0f + 1e-5f) << "v=" << v << " T=" << T << " R=" << R;
            }
        }
    }
}

TEST(DPMonotonicityTest, UpperEVNonnegativeFromEmpty) {
    const DPTables& dp = shared_dp();
    int8_t Sc[6] = {0, 0, 0, 0, 0, 0};
    for (int v : kVariants) {
        for (int T = 6; T <= 78; ++T) {
            float ev = get_upper_ev(dp, static_cast<Variant>(v), Sc, T, 0);
            EXPECT_GE(ev, 0.0f) << "v=" << v << " T=" << T;
        }
    }
}

TEST(DPMonotonicityTest, MiddleAndLowerInRangeFromEmpty) {
    const DPTables& dp = shared_dp();
    int8_t mid[3] = {0, 0, 0};
    int8_t low[5] = {0, 0, 0, 0, 0};
    for (int v : kVariants) {
        for (int T = 2; T <= 78; ++T) {
            float pm = get_middle_prob(dp, static_cast<Variant>(v), mid, T);
            float em = get_middle_ev  (dp, static_cast<Variant>(v), mid, T);
            EXPECT_GE(pm, 0.0f); EXPECT_LE(pm, 1.0f + 1e-5f);
            EXPECT_GE(em, 0.0f);
        }
        for (int T = 5; T <= 78; ++T) {
            float pl = get_lower_prob(dp, static_cast<Variant>(v), low, T);
            float el = get_lower_ev  (dp, static_cast<Variant>(v), low, T);
            EXPECT_GE(pl, 0.0f); EXPECT_LE(pl, 1.0f + 1e-5f);
            EXPECT_GE(el, 0.0f);
        }
    }
}

TEST(DPMonotonicityTest, MoreTurnsHelpEmptyUpper) {
    // For an empty upper section under FREE, probability of reaching R and
    // expected EV should be non-decreasing in T (more turns can't hurt).
    const DPTables& dp = shared_dp();
    int8_t Sc[6] = {0, 0, 0, 0, 0, 0};
    for (int R : {30, 60, 80}) {
        float prev = -1.0f;
        for (int T = 6; T <= 78; ++T) {  // need at least 6 turns to fill 6 cells
            float p = get_upper_prob(dp, Variant::FREE, Sc, T, R);
            EXPECT_GE(p, prev - 1e-5f) << "R=" << R << " T=" << T
                                       << " prev=" << prev << " p=" << p;
            prev = p;
        }
    }
    float prev_ev = -1.0f;
    for (int T = 6; T <= 78; ++T) {
        float ev = get_upper_ev(dp, Variant::FREE, Sc, T, 0);
        EXPECT_GE(ev, prev_ev - 1e-3f) << "T=" << T;
        prev_ev = ev;
    }
}

TEST(DPMonotonicityTest, MoreTurnsHelpEmptyMiddleLower) {
    const DPTables& dp = shared_dp();
    int8_t mid[3] = {0, 0, 0};
    int8_t low[5] = {0, 0, 0, 0, 0};
    float prev_pm = -1.0f, prev_em = -1.0f;
    for (int T = 2; T <= 78; ++T) {
        float pm = get_middle_prob(dp, Variant::FREE, mid, T);
        float em = get_middle_ev  (dp, Variant::FREE, mid, T);
        EXPECT_GE(pm, prev_pm - 1e-5f) << "T=" << T;
        EXPECT_GE(em, prev_em - 1e-3f) << "T=" << T;
        prev_pm = pm; prev_em = em;
    }
    float prev_pl = -1.0f, prev_el = -1.0f;
    for (int T = 5; T <= 78; ++T) {
        float pl = get_lower_prob(dp, Variant::FREE, low, T);
        float el = get_lower_ev  (dp, Variant::FREE, low, T);
        EXPECT_GE(pl, prev_pl - 1e-5f) << "T=" << T;
        EXPECT_GE(el, prev_el - 1e-3f) << "T=" << T;
        prev_pl = pl; prev_el = el;
    }
}

TEST(DPMonotonicityTest, ProbReachZeroIsOne) {
    // Reaching cumulative target R=0 is trivially guaranteed (no requirement).
    const DPTables& dp = shared_dp();
    int8_t Sc[6] = {0, 0, 0, 0, 0, 0};
    for (int v : kVariants) {
        for (int T = 6; T <= 78; ++T) {
            float p = get_upper_prob(dp, static_cast<Variant>(v), Sc, T, 0);
            EXPECT_NEAR(p, 1.0f, 1e-4f) << "v=" << v << " T=" << T;
        }
    }
}
