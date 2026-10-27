// Copyright 2026 WheelOS. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gtest/gtest.h"

#include "spconv/sparse_backbone.h"

namespace wheelos::perception::ops::spconv {
namespace {

uint16_t HalfBits(float value) {
  const __half half = __float2half(value);
  uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

SparseModel Model(bool xyz) {
  SparseModel model;
  model.input = "input";
  model.output = "output";
  model.input_channels = 8;
  model.output_shape = {1, 24, 8};
  SparseModelLayer conv;
  conv.op = "SparseConvolution";
  conv.inputs = {"input"};
  conv.output = "conv";
  conv.kernel = conv.stride = conv.dilation = {1, 1, 1};
  conv.padding = {0, 0, 0};
  conv.input_grid = conv.output_grid = {2, 3, 4};
  conv.weight_shape = {8, 1, 1, 1, 8};
  conv.weight_data.resize(64, HalfBits(0));
  for (int channel = 0; channel < 8; ++channel) {
    conv.weight_data[channel * 8 + channel] = HalfBits(1);
  }
  conv.bias_data.resize(8, HalfBits(1));
  conv.bound = 8;
  conv.subm = true;
  model.layers.push_back(conv);
  SparseModelLayer add;
  add.op = "Add";
  add.inputs = {"conv", "conv"};
  add.output = "sum";
  model.layers.push_back(add);
  SparseModelLayer relu;
  relu.op = "Relu";
  relu.inputs = {"sum"};
  relu.output = "relu";
  model.layers.push_back(relu);
  SparseModelLayer scatter;
  scatter.op = "ScatterDense";
  scatter.inputs = {"relu"};
  scatter.output = "dense";
  scatter.input_grid = {2, 3, 4};
  scatter.dense_shape = {1, 8, 2, 3, 4};
  scatter.xyz = xyz;
  model.layers.push_back(scatter);
  SparseModelLayer transpose;
  transpose.op = "Transpose";
  transpose.inputs = {"dense"};
  transpose.output = "transposed";
  transpose.transpose = {0, 2, 3, 4, 1};
  transpose.dense_shape = {1, 2, 3, 4, 8};
  model.layers.push_back(transpose);
  SparseModelLayer reshape;
  reshape.op = "Reshape";
  reshape.inputs = {"transposed"};
  reshape.output = "output";
  reshape.dense_shape = {1, 24, 8};
  model.layers.push_back(reshape);
  return model;
}

class SparseBackboneTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaMalloc(&features_, 16 * sizeof(uint16_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&indices_, 8 * sizeof(int32_t)), cudaSuccess);
    for (auto& stream : streams_) {
      ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                cudaSuccess);
    }
  }

  void TearDown() override {
    for (auto stream : streams_) {
      if (stream != nullptr) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      }
    }
    EXPECT_EQ(cudaFree(features_), cudaSuccess);
    EXPECT_EQ(cudaFree(indices_), cudaSuccess);
  }

  void Compare(SparseBackbone* backbone, int count, cudaStream_t stream) {
    std::vector<uint16_t> features(count * 8);
    std::vector<uint16_t> expected(24 * 8, HalfBits(0));
    const std::array<int32_t, 8> indices = {0, 0, 1, 2, 0, 1, 2, 3};
    for (int point = 0; point < count; ++point) {
      const int spatial =
          (indices[point * 4 + 1] * 3 + indices[point * 4 + 2]) * 4 +
          indices[point * 4 + 3];
      for (int channel = 0; channel < 8; ++channel) {
        const float value = channel - 3 + point;
        features[point * 8 + channel] = HalfBits(value);
        expected[spatial * 8 + channel] =
            HalfBits(value + 1 > 0 ? 2 * (value + 1) : 0);
      }
    }
    ASSERT_EQ(cudaMemcpyAsync(features_, features.data(),
                              features.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(indices_, indices.data(), count * 4 * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream),
        cudaSuccess);
    void* output =
        backbone->Forward(features_, indices_, count, {2, 3, 4}, stream);
    std::vector<uint16_t> actual(expected.size());
    ASSERT_EQ(
        cudaMemcpyAsync(actual.data(), output, actual.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(actual, expected);
  }

  void* features_ = nullptr;
  void* indices_ = nullptr;
  std::array<cudaStream_t, 2> streams_{};
};

TEST_P(SparseBackboneTest, MatchesCpuReferenceAcrossCountsAndStreams) {
  auto backbone = CreateSparseBackbone(Model(GetParam()));
  EXPECT_EQ(backbone->Shape(), (std::vector<int64_t>{1, 24, 8}));
  Compare(backbone.get(), 2, streams_[0]);
  Compare(backbone.get(), 1, streams_[1]);
  Compare(backbone.get(), 2, streams_[0]);
  EXPECT_THROW(
      backbone->Forward(features_, indices_, 0, {2, 3, 4}, streams_[0]),
      std::runtime_error);
  EXPECT_THROW(
      backbone->Forward(features_, indices_, 1, {4, 3, 2}, streams_[0]),
      std::runtime_error);
  Compare(backbone.get(), 1, streams_[1]);
}

INSTANTIATE_TEST_SUITE_P(CoordinateOrder, SparseBackboneTest,
                         testing::Values(false, true));

}  // namespace
}  // namespace wheelos::perception::ops::spconv
