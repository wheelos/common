/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "modules/common/kv_db/kv_db.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <optional>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

DECLARE_string(kv_db_path);

namespace apollo {
namespace common {

class KVDBTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_db_path_ = FLAGS_kv_db_path;
    test_root_ = std::filesystem::temp_directory_path() /
                 ("apollo_kv_db_test_" +
                  std::to_string(std::chrono::steady_clock::now()
                                     .time_since_epoch()
                                     .count()));
    std::filesystem::create_directories(test_root_);
    FLAGS_kv_db_path = (test_root_ / "kv_db.sqlite").string();
  }

  void TearDown() override {
    FLAGS_kv_db_path = original_db_path_;
    std::error_code error;
    std::filesystem::remove_all(test_root_, error);
  }

  std::filesystem::path test_root_;
  std::string original_db_path_;
};

TEST_F(KVDBTest, CRUD) {
  EXPECT_TRUE(KVDB::Delete("test_key"));
  EXPECT_FALSE(KVDB::Get("test_key").has_value());
  // Put
  EXPECT_TRUE(KVDB::Put("test_key", "val0"));
  EXPECT_EQ("val0", KVDB::Get("test_key").value());

  // Update
  EXPECT_TRUE(KVDB::Put("test_key", "val1"));
  EXPECT_EQ("val1", KVDB::Get("test_key").value());

  // Delete
  EXPECT_TRUE(KVDB::Delete("test_key"));
  EXPECT_FALSE(KVDB::Get("test_key").has_value());
}

TEST_F(KVDBTest, CheckedGetDistinguishesMissingEmptyAndFailure) {
  std::optional<std::string> value;
  ASSERT_TRUE(KVDB::Get("missing", &value));
  EXPECT_FALSE(value.has_value());

  ASSERT_TRUE(KVDB::Put("empty", ""));
  ASSERT_TRUE(KVDB::Get("empty", &value));
  ASSERT_TRUE(value.has_value());
  EXPECT_TRUE(value->empty());
  EXPECT_FALSE(KVDB::Get("empty").has_value());

  EXPECT_FALSE(KVDB::Get("empty", nullptr));
  FLAGS_kv_db_path = (test_root_ / "missing" / "kv_db.sqlite").string();
  EXPECT_FALSE(KVDB::Get("missing", &value));
  EXPECT_FALSE(value.has_value());
  FLAGS_kv_db_path = (test_root_ / "kv_db.sqlite").string();
}

TEST_F(KVDBTest, BindsKeysAndValuesAsData) {
  const std::string key = "key' OR 1=1 --";
  const std::string value = "value'); DELETE FROM key_value; --";
  ASSERT_TRUE(KVDB::Put(key, value));
  EXPECT_EQ(value, KVDB::Get(key).value());
  EXPECT_FALSE(KVDB::Get("key").has_value());
}

TEST_F(KVDBTest, ConcurrentOperationsReportSuccess) {
  constexpr int kThreadCount = 10;
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int index = 0; index < kThreadCount; ++index) {
    threads.emplace_back([index, &succeeded]() {
      const std::string key = "thread_" + std::to_string(index);
      std::optional<std::string> value;
      if (!KVDB::Put(key, "value") || !KVDB::Get(key, &value) ||
          !value.has_value() || *value != "value" || !KVDB::Delete(key)) {
        succeeded.store(false);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_TRUE(succeeded.load());
}

}  // namespace common
}  // namespace apollo
