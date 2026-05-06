import re

with open("src/engine/tensor.cc", "r") as f:
    code = f.read()

# We want to replace the whole generate_tensor body, from `void generate_tensor(` to the `assert(idx == kTensorSize); \n}`.

start_str = "void generate_tensor(const BoardState& board, const GameContext& ctx,"
end_str = "    assert(idx == kTensorSize);\n}"

start_idx = code.find(start_str)
end_idx = code.find(end_str, start_idx) + len(end_str)

new_body = """void generate_tensor(const BoardState& board, const GameContext& ctx,
                     int player, const PrecomputedTables& tables,
                     float* out) {
    const int opp = 1 - player;
    const DPTables& dp = tables.dp_tables;
    const bool dp_ready = !dp.dp_t1.empty();

    int idx = 0;

    struct PCData {
        int empty_in_col;
        int upper_sum;
        float e_raw;
        int raw_score;
        int pot_score;
        float p_one[13]; // kNumRows
        float grp_E[18];
        float grp_F[15];
    };
    PCData pc[2][6];

    static const int kUpperTargets[5] = {60, 70, 80, 90, 100};

    for (int pi = 0; pi < kNumPlayers; ++pi) {
        int p = (pi == 0) ? player : opp;
        int player_filled = count_filled_cells(board, p);

        for (int col = 0; col < kNumColumns; ++col) {
            PCData& d = pc[pi][col];
            
            d.upper_sum = ctx.upper_sum[p][col];
            d.raw_score = compute_column_raw_score(board, ctx, p, col);
            d.pot_score = compute_column_potential_score(board, ctx, p, col);

            // Group C precomputation
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v != kCellEmpty) {
                    d.p_one[row] = 1.0f;
                    continue;
                }
                if (row == kRowSS && ctx.ls_scratched[p][col]) {
                    d.p_one[row] = 0.0f; continue;
                }
                if (row == kRowLS && ctx.ss_scratched[p][col]) {
                    d.p_one[row] = 0.0f; continue;
                }

                int golden_min = ctx.golden_max[col][row];
                if (golden_min == 0) golden_min = 1;

                if (row == kRowSS) {
                    int8_t ls_val = board.cells[p][col][kRowLS];
                    if (ls_val != kCellEmpty && ls_val > 0 && golden_min >= ls_val) {
                        d.p_one[row] = 0.0f; continue;
                    }
                }
                if (row == kRowLS) {
                    int max_ss = ctx.golden_max[col][kRowSS];
                    if (max_ss > 0) golden_min = std::max(golden_min, max_ss + 1);
                }

                int max_for_row = kMaxScorePerRow[row];
                if (golden_min > max_for_row) {
                    d.p_one[row] = 0.0f; continue;
                }
                int thresh = std::min(golden_min, 100);
                float p_one = (col == kColTurbo)
                    ? tables.prob_tables.prob_2rolls_compound[row][thresh][1]
                    : tables.prob_tables.prob_3rolls_compound[row][thresh][1];
                d.p_one[row] = p_one;
            }

            int8_t Sc_U[6], Sc_M[2], Sc_L[5];
            int EU, EM, EL;
            build_Sc(p, col, board, ctx, Sc_U, Sc_M, Sc_L, EU, EM, EL);
            d.empty_in_col = EU + EM + EL;

            int T_min = d.empty_in_col;
            int T_max = std::max(T_min, 78 - player_filled);
            int T_mid = (T_min + T_max) / 2;
            int Ts[3] = {T_min, T_mid, T_max};

            int cur_upper = ctx.upper_sum[p][col];
            int R_clean = std::max(0, 60 - cur_upper);
            bool low_scratch = ctx.lower_has_scratch[p][col];
            Variant v = get_variant(col);

            int e_idx = 0;
            int f_idx = 0;
            float eu_min = 0, em_min = 0, el_min = 0;

            for (int ti = 0; ti < 3; ++ti) {
                int T = Ts[ti];
                int TU, TM, TL;
                apportion_turns(T, EU, EM, EL, TU, TM, TL);

                if (dp_ready) {
                    for (int k = 0; k < 5; ++k) {
                        int R = kUpperTargets[k] - cur_upper;
                        if (R < 0) R = 0;
                        d.grp_E[e_idx++] = get_upper_prob(dp, v, Sc_U, TU, R);
                    }
                    float eu = get_upper_ev(dp, v, Sc_U, TU, cur_upper);
                    d.grp_E[e_idx++] = std::min(1.0f, std::max(0.0f, eu / 100.0f));

                    float p_mid = get_middle_prob(dp, v, Sc_M, TM);
                    float ev_mid = get_middle_ev (dp, v, Sc_M, TM);
                    float p_low = get_lower_prob (dp, v, Sc_L, TL);
                    float ev_low = get_lower_ev  (dp, v, Sc_L, TL);
                    float p_up_60 = get_upper_prob(dp, v, Sc_U, TU, R_clean);

                    d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, p_mid));
                    d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, ev_mid / 60.0f));
                    d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, p_low));
                    d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, ev_low / 250.0f));
                    float p_clean = low_scratch ? 0.0f : (p_up_60 * p_mid * p_low);
                    d.grp_F[f_idx++] = std::min(1.0f, std::max(0.0f, p_clean));

                    if (ti == 0) {
                        eu_min = eu;
                        em_min = ev_mid;
                        el_min = ev_low;
                    }
                } else {
                    for(int i=0; i<6; ++i) d.grp_E[e_idx++] = 0.0f;
                    for(int i=0; i<5; ++i) d.grp_F[f_idx++] = 0.0f;
                }
            }

            int filled_score = 0;
            for (int r = 6; r <= 12; ++r) {
                int8_t cv = board.cells[p][col][r];
                if (cv > 0) filled_score += cv;
            }
            d.e_raw = eu_min + em_min + el_min + static_cast<float>(filled_score);
        }
    }

    // Group A (312)
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        int p = (pi == 0) ? player : opp;
        for (int col = 0; col < kNumColumns; ++col) {
            for (int row = 0; row < kNumRows; ++row) {
                int8_t v = board.cells[p][col][row];
                if (v == kCellEmpty) {
                    out[idx++] = 0.0f;
                    out[idx++] = 0.0f;
                } else {
                    out[idx++] = 1.0f;
                    out[idx++] = static_cast<float>(v) / static_cast<float>(kMaxScorePerRow[row]);
                }
            }
        }
    }

    // Group B.1
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        for (int col = 0; col < kNumColumns; ++col) {
            out[idx++] = std::min(1.0f, static_cast<float>(pc[pi][col].upper_sum) / 100.0f);
            out[idx++] = std::min(1.0f, std::max(0.0f, pc[pi][col].e_raw / 500.0f));
        }
    }

    // Group B.2
    long long total_duel_now = 0;
    double total_duel_E = 0.0;
    for (int col = 0; col < kNumColumns; ++col) {
        int coeff = board.coefficients[col];
        int rem_me = pc[0][col].empty_in_col;
        int rem_opp = pc[1][col].empty_in_col;
        out[idx++] = static_cast<float>(rem_me) / static_cast<float>(kNumRows);
        out[idx++] = static_cast<float>(rem_opp) / static_cast<float>(kNumRows);

        int my_r = pc[0][col].raw_score;
        int opp_r = pc[1][col].raw_score;
        int crush_my = crush_multiplier(my_r, opp_r);
        int crush_opp = crush_multiplier(opp_r, my_r);
        int active_crush = std::max(crush_my, crush_opp);
        long long margin_now = static_cast<long long>(my_r - opp_r) * coeff * active_crush;
        out[idx++] = std::tanh(static_cast<float>(margin_now) / 15000.0f);
        total_duel_now += margin_now;

        int my_eR = static_cast<int>(pc[0][col].e_raw + 0.5f);
        int opp_eR = static_cast<int>(pc[1][col].e_raw + 0.5f);
        int crush_my_E = crush_multiplier(my_eR, opp_eR);
        int crush_opp_E = crush_multiplier(opp_eR, my_eR);
        int active_crush_E = std::max(crush_my_E, crush_opp_E);
        double margin_E = static_cast<double>(pc[0][col].e_raw - pc[1][col].e_raw) * coeff * active_crush_E;
        out[idx++] = std::tanh(static_cast<float>(margin_E / 15000.0));
        total_duel_E += margin_E;

        out[idx++] = static_cast<float>(crush_my - crush_opp) / 5.0f;
        out[idx++] = static_cast<float>(crush_my_E - crush_opp_E) / 5.0f;

        const float curr_scales[4] = {500.0f, 750.0f, 1000.0f, 1250.0f};
        for (int k = 0; k < 4; ++k) {
            out[idx++] = pts_to_Nx(k + 2, static_cast<float>(my_r), static_cast<float>(opp_r), curr_scales[k]);
        }
        for (int k = 0; k < 4; ++k) {
            out[idx++] = pts_to_Nx(k + 2, pc[0][col].e_raw, pc[1][col].e_raw, curr_scales[k]);
        }
    }

    // Group C
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        for (int col = 0; col < kNumColumns; ++col) {
            for (int row = 0; row < kNumRows; ++row) {
                out[idx++] = pc[pi][col].p_one[row];
            }
        }
    }

    // Group D
    for (int col = 0; col < kNumColumns; ++col) {
        out[idx++] = static_cast<float>(board.coefficients[col]) / 18.0f;
    }
    out[idx++] = static_cast<float>(board.cells_filled) / static_cast<float>(kTotalCells);
    out[idx++] = std::tanh(static_cast<float>(total_duel_now) / 80000.0f);
    out[idx++] = std::tanh(static_cast<float>(total_duel_E) / 100000.0);

    int my_min = 0, my_max = 0, opp_min = 0, opp_max = 0;
    for (int col = 0; col < kNumColumns; ++col) {
        my_min += pc[0][col].raw_score;
        my_max += pc[0][col].pot_score;
        opp_min += pc[1][col].raw_score;
        opp_max += pc[1][col].pot_score;
    }
    out[idx++] = (my_min > opp_max) ? 1.0f : 0.0f;
    out[idx++] = (my_max < opp_min) ? 1.0f : 0.0f;

    int filled = static_cast<int>(board.cells_filled);
    out[idx++] = (filled > 50)  ? 1.0f : 0.0f;
    out[idx++] = (filled > 100) ? 1.0f : 0.0f;
    out[idx++] = (filled > 140) ? 1.0f : 0.0f;

    // Group E
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        for (int col = 0; col < kNumColumns; ++col) {
            for (int i = 0; i < 18; ++i) {
                out[idx++] = pc[pi][col].grp_E[i];
            }
        }
    }

    // Group F
    for (int pi = 0; pi < kNumPlayers; ++pi) {
        for (int col = 0; col < kNumColumns; ++col) {
            for (int i = 0; i < 15; ++i) {
                out[idx++] = pc[pi][col].grp_F[i];
            }
        }
    }

    assert(idx == kTensorSize);
}"""

with open("src/engine/tensor.cc", "w") as f:
    f.write(code[:start_idx] + new_body + code[end_idx:])
