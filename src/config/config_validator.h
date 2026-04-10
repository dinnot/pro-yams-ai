#pragma once

#include <string>
#include <vector>

#include "config/app_config.h"

// ---------------------------------------------------------------------------
// ValidationResult — outcome of validate_config().
// ---------------------------------------------------------------------------
struct ValidationResult {
    bool                     ok = true;
    std::vector<std::string> errors;

    void fail(std::string msg) {
        ok = false;
        errors.push_back(std::move(msg));
    }
};

/// Validate all fields in `cfg`.  Returns a ValidationResult whose `ok`
/// field is true iff no constraint violations were found.
ValidationResult validate_config(const AppConfig& cfg);
