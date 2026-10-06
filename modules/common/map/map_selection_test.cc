// Copyright 2026 WheelOS. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "modules/common/map/map_selection.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "modules/common/kv_db/kv_db.h"

DECLARE_string(kv_db_path);

namespace apollo {
namespace common {
namespace {

class MapSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_db_path_ = FLAGS_kv_db_path;
    const char* asset_root = std::getenv("WHEELOS_ASSET_ROOT");
    if (asset_root != nullptr) {
      original_asset_root_ = asset_root;
    }
    test_root_ =
        std::filesystem::temp_directory_path() /
        ("apollo_map_selection_test_" +
         std::to_string(std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count()));
    const auto map_directory = test_root_ / "assets/sites/map/borregas_ave";
    std::filesystem::create_directories(map_directory);
    FLAGS_kv_db_path = (test_root_ / "selected_map.sqlite").string();
    MapSelection::ClearTestMapDirectory();
    ASSERT_EQ(0, setenv("WHEELOS_ASSET_ROOT",
                        (test_root_ / "assets").c_str(), 1));
    ASSERT_TRUE(KVDB::Delete("runtime/selected_map"));
  }

  void TearDown() override {
    FLAGS_kv_db_path = original_db_path_;
    if (original_asset_root_.has_value()) {
      setenv("WHEELOS_ASSET_ROOT", original_asset_root_->c_str(), 1);
    } else {
      unsetenv("WHEELOS_ASSET_ROOT");
    }
    MapSelection::ClearTestMapDirectory();
    std::error_code error;
    std::filesystem::remove_all(test_root_, error);
  }

  std::filesystem::path test_root_;
  std::string original_db_path_;
  std::optional<std::string> original_asset_root_;
};

TEST_F(MapSelectionTest, UsesDefaultWhenNoMapWasPersisted) {
  SelectedMap selected_map;
  ASSERT_TRUE(MapSelection::GetSelectedMap(&selected_map));
  EXPECT_EQ("borregas_ave", selected_map.id);
  EXPECT_EQ((test_root_ / "assets/sites/map/borregas_ave").string(),
            selected_map.directory);
}

TEST_F(MapSelectionTest, PersistsAndResolvesSelectedMap) {
  std::filesystem::create_directories(test_root_ / "assets/sites/map/map_v2");
  ASSERT_TRUE(MapSelection::SelectMap("map_v2"));

  SelectedMap selected_map;
  ASSERT_TRUE(MapSelection::GetSelectedMap(&selected_map));
  EXPECT_EQ("map_v2", selected_map.id);
  EXPECT_EQ((test_root_ / "assets/sites/map/map_v2").string(),
            selected_map.directory);
}

TEST_F(MapSelectionTest, RejectsPathTraversal) {
  std::string map_directory;
  EXPECT_FALSE(MapSelection::ResolveMap("../outside", &map_directory));
  EXPECT_FALSE(MapSelection::SelectMap("../outside"));
  EXPECT_FALSE(MapSelection::ResolveMap("/absolute/path", &map_directory));
  EXPECT_FALSE(MapSelection::ResolveMap("map/submap", &map_directory));
}

TEST_F(MapSelectionTest, RejectsMissingMapWithoutChangingSelection) {
  ASSERT_TRUE(KVDB::Put("runtime/selected_map", "borregas_ave"));
  EXPECT_FALSE(MapSelection::SelectMap("missing_map"));

  std::optional<std::string> stored_map_id;
  ASSERT_TRUE(KVDB::Get("runtime/selected_map", &stored_map_id));
  ASSERT_TRUE(stored_map_id.has_value());
  EXPECT_EQ("borregas_ave", *stored_map_id);
}

TEST_F(MapSelectionTest, RejectsInvalidPersistedMapId) {
  ASSERT_TRUE(KVDB::Put("runtime/selected_map", "../outside"));
  SelectedMap selected_map;
  EXPECT_FALSE(MapSelection::GetSelectedMap(&selected_map));
}

}  // namespace
}  // namespace common
}  // namespace apollo
