#pragma once

#include <iostream>
#include <ostream>

#include "config/app_config.h"

/// Print all AppConfig fields in YAML-compatible format to `out`.
void print_config(const AppConfig& cfg, std::ostream& out = std::cout);
