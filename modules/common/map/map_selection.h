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

#pragma once

#include <string>

namespace apollo {
namespace common {

struct SelectedMap {
  std::string id;
  std::string directory;
};

class MapSelection {
 public:
  static bool GetSelectedMap(SelectedMap* selected_map);
  static bool ResolveMap(const std::string& map_id, std::string* map_directory);
  static bool SelectMap(const std::string& map_id);

  static void SetTestMapDirectory(const std::string& map_directory);
  static void ClearTestMapDirectory();
};

}  // namespace common
}  // namespace apollo
