#pragma once

#include <iostream>
#include <ostream>

#include "config/app_config.h"

/// Print all AppConfig fields in YAML-compatible format to `out`.
void print_config(const AppConfig& cfg, std::ostream& out = std::cout);

/// Save all AppConfig fields to a YAML file at `path`.
void save_config(const AppConfig& cfg, const std::string& path);
