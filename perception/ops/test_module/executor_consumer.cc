// Copyright 2026 WheelOS. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <utility>

#include "spconv/sparse_backbone.h"

std::unique_ptr<wheelos::perception::ops::spconv::SparseBackbone>
CreateExecutor(wheelos::perception::ops::spconv::SparseModel model) {
  return wheelos::perception::ops::spconv::CreateSparseBackbone(
      std::move(model));
}
