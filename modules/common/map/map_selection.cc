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

#include <cctype>
#include <filesystem>
#include <mutex>
#include <system_error>

#include "cyber/common/log.h"
#include "cyber/common/resource_manager.h"
#include "modules/common/kv_db/kv_db.h"

namespace apollo {
namespace common {
namespace {

constexpr char kSelectedMapKey[] = "runtime/selected_map";
constexpr char kDefaultMapId[] = "borregas_ave";
std::mutex test_map_directory_mutex;
std::string test_map_directory;

bool IsValidMapId(const std::string& map_id) {
  if (map_id.empty() || map_id == "." || map_id == "..") {
    return false;
  }
  for (const unsigned char character : map_id) {
    if (!std::isalnum(character) && character != '_' && character != '-' &&
        character != '.') {
      return false;
    }
  }
  return true;
}

bool ResolveExplicitMapDirectory(const std::string& path,
                                 SelectedMap* selected_map) {
  std::error_code error;
  const std::filesystem::path canonical_path =
      std::filesystem::canonical(path, error);
  if (error || !std::filesystem::is_directory(canonical_path, error) || error) {
    AERROR << "Explicit map directory is not an accessible directory: "
           << path;
    return false;
  }
  std::filesystem::directory_iterator iterator(canonical_path, error);
  if (error) {
    AERROR << "Cannot read explicit map directory: " << canonical_path;
    return false;
  }
  selected_map->id = canonical_path.filename().string();
  selected_map->directory = canonical_path.string();
  AINFO << "[MAP] selected " << selected_map->id
        << " from test map directory override: "
        << selected_map->directory;
  return true;
}

}  // namespace

bool MapSelection::GetSelectedMap(SelectedMap* selected_map) {
  if (selected_map == nullptr) {
    AERROR << "Cannot get selected map: output is null.";
    return false;
  }
  *selected_map = {};

  std::string test_directory;
  {
    std::lock_guard<std::mutex> lock(test_map_directory_mutex);
    test_directory = test_map_directory;
  }
  if (!test_directory.empty()) {
    return ResolveExplicitMapDirectory(test_directory, selected_map);
  }

  std::optional<std::string> stored_map_id;
  if (!KVDB::Get(kSelectedMapKey, &stored_map_id)) {
    AERROR << "Failed to read persisted map selection.";
    return false;
  }
  const std::string map_id =
      stored_map_id.has_value() ? *stored_map_id : kDefaultMapId;
  if (!IsValidMapId(map_id)) {
    AERROR << "Persisted map ID is invalid: " << map_id;
    return false;
  }
  if (!ResolveMap(map_id, &selected_map->directory)) {
    return false;
  }
  selected_map->id = map_id;
  AINFO << "[MAP] selected " << map_id << " from "
        << (stored_map_id.has_value() ? "KVDB" : "default") << ": "
        << selected_map->directory;
  return true;
}

bool MapSelection::ResolveMap(const std::string& map_id,
                              std::string* map_directory) {
  if (map_directory == nullptr) {
    AERROR << "Cannot resolve map: output is null.";
    return false;
  }
  map_directory->clear();
  if (!IsValidMapId(map_id)) {
    AERROR << "Map ID must be a single safe path component: " << map_id;
    return false;
  }
  const std::string asset_key = "sites/map/" + map_id;
  if (!cyber::common::ResourceManager::ResolveAssetPath(asset_key,
                                                         map_directory)) {
    AERROR << "Failed to resolve map asset: " << asset_key;
    return false;
  }
  return true;
}

bool MapSelection::SelectMap(const std::string& map_id) {
  std::string map_directory;
  if (!ResolveMap(map_id, &map_directory)) {
    return false;
  }
  if (!KVDB::Put(kSelectedMapKey, map_id)) {
    AERROR << "Failed to persist selected map ID: " << map_id;
    return false;
  }
  AINFO << "[MAP] persisted selection " << map_id << ": " << map_directory;
  return true;
}

void MapSelection::SetTestMapDirectory(const std::string& map_directory) {
  std::lock_guard<std::mutex> lock(test_map_directory_mutex);
  test_map_directory = map_directory;
}

void MapSelection::ClearTestMapDirectory() {
  std::lock_guard<std::mutex> lock(test_map_directory_mutex);
  test_map_directory.clear();
}

}  // namespace common
}  // namespace apollo
