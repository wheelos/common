// Copyright 2026 WheelOS. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wheelos::perception::ops::spconv {

struct SparseModelLayer {
  std::string op;
  std::vector<std::string> inputs;
  std::string output;
  std::vector<int> kernel, stride, padding, dilation, input_grid, output_grid;
  std::vector<int> dense_shape;
  std::vector<int> transpose;
  std::vector<int64_t> weight_shape;
  std::vector<uint16_t> weight_data, bias_data;
  int bound = 0;
  bool subm = false;
  bool relu = false;
  bool xyz = false;
};

struct SparseModel {
  std::string input, output;
  int input_channels = 0;
  std::vector<int64_t> output_shape;
  std::vector<SparseModelLayer> layers;
};

}  // namespace wheelos::perception::ops::spconv
