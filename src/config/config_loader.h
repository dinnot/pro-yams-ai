#pragma once

#include <string>
#include "config/app_config.h"

/// Load AppConfig from a YAML file at `path`.
/// Throws std::runtime_error if the file cannot be read or parsed.
AppConfig load_config(const std::string& path);

/// Load config from file (if --config given) and apply CLI overrides.
AppConfig load_config(int argc, char* argv[]);

/// Apply command-line overrides to an existing config.
void apply_cli_overrides(AppConfig& config, int argc, char* argv[]);

/// Return an AppConfig populated with all compiled-in defaults.
AppConfig default_config();
