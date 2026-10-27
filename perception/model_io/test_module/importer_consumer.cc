// Copyright 2026 WheelOS. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "wheelos/perception/model_io/sparse_model.h"

wheelos::perception::ops::spconv::SparseModel ImportModel(
    const std::string& path) {
  return wheelos::perception::model_io::LoadSparseModel(path);
}
