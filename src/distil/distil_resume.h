#pragma once

#include <string>

// Forward declare to avoid pulling in libtorch headers here.
template <typename Traits> class DistilLoopT;

// ---------------------------------------------------------------------------
// resume_distil_from_checkpoint — restore a DistilLoopT's persistent state
// from the latest checkpoint found in `dir`.
//
// Restored:  student model weights, Adam optimizer state, training_step.
// Reset:     consecutive_passes_, rolling_mse_. Both rebuild quickly from
//            subsequent eval / train_step calls — they're transient enough
//            that re-saving them in the checkpoint isn't worth the schema
//            change.
//
// Refuses the resume if the .optimizer sibling is missing — restarting Adam
// on already-trained weights can spike loss enough to undo prior progress.
// In that case the caller should use --checkpoint <stem> instead, which
// loads only weights and accepts the fresh-optimizer trade-off explicitly.
//
// @return true on success, false if no checkpoint files are present in dir.
// @throws std::runtime_error if checkpoints exist but the .optimizer
//         sibling is missing.
// ---------------------------------------------------------------------------
template <typename Traits>
bool resume_distil_from_checkpoint(DistilLoopT<Traits>& loop,
                                    const std::string& dir);

// ---------------------------------------------------------------------------
// init_distil_from_checkpoint — load only model weights from `path` (which
// may be a checkpoint stem, e.g. ".../checkpoint_step_5000", or a directory
// in which case the latest stem is used). Training restarts from step 0
// with a fresh Adam optimizer.
// ---------------------------------------------------------------------------
template <typename Traits>
bool init_distil_from_checkpoint(DistilLoopT<Traits>& loop,
                                  const std::string& path);
