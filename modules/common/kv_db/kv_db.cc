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

#include <limits>
#include <sqlite3.h>

#include "gflags/gflags.h"

#include "cyber/common/log.h"

DEFINE_string(kv_db_path, "/apollo/data/kv_db.sqlite",
              "Path to Key-value DB file.");

namespace apollo {
namespace common {
namespace {

// Self-maintained sqlite instance.
class SqliteWraper {
 public:
  SqliteWraper() {
    if (sqlite3_open(FLAGS_kv_db_path.c_str(), &db_) != 0) {
      AERROR << "Can't open Key-Value database: "
             << (db_ == nullptr ? "unknown SQLite error"
                                : sqlite3_errmsg(db_));
      Release();
      return;
    }
    if (sqlite3_busy_timeout(db_, 5000) != SQLITE_OK) {
      AERROR << "Failed to configure Key-Value database busy timeout: "
             << sqlite3_errmsg(db_);
      Release();
      return;
    }

    static const char *kCreateTableSql =
        "CREATE TABLE IF NOT EXISTS key_value "
        "(key VARCHAR(128) PRIMARY KEY NOT NULL, value TEXT);";
    char *error = nullptr;
    if (sqlite3_exec(db_, kCreateTableSql, nullptr, nullptr, &error) !=
        SQLITE_OK) {
      AERROR << "Failed to initialize Key-Value database: "
             << (error == nullptr ? sqlite3_errmsg(db_) : error);
      sqlite3_free(error);
      Release();
    }
  }

  ~SqliteWraper() { Release(); }

  bool Put(std::string_view key, std::string_view value) {
    sqlite3_stmt *statement = nullptr;
    if (!Prepare("INSERT OR REPLACE INTO key_value (key, value) VALUES (?, ?);",
                 &statement)) {
      return false;
    }

    if (!BindText(statement, 1, key) || !BindText(statement, 2, value)) {
      sqlite3_finalize(statement);
      return false;
    }

    const bool success = StepDone(statement);
    sqlite3_finalize(statement);
    return success;
  }

  bool Delete(std::string_view key) {
    sqlite3_stmt *statement = nullptr;
    if (!Prepare("DELETE FROM key_value WHERE key = ?;", &statement)) {
      return false;
    }

    if (!BindText(statement, 1, key)) {
      sqlite3_finalize(statement);
      return false;
    }

    const bool success = StepDone(statement);
    sqlite3_finalize(statement);
    return success;
  }

  bool Get(std::string_view key, std::optional<std::string> *value) {
    if (value == nullptr) {
      AERROR << "Cannot read Key-Value database into a null output.";
      return false;
    }
    value->reset();

    sqlite3_stmt *statement = nullptr;
    if (!Prepare("SELECT value FROM key_value WHERE key = ?;", &statement)) {
      return false;
    }
    if (!BindText(statement, 1, key)) {
      sqlite3_finalize(statement);
      return false;
    }

    const int result = sqlite3_step(statement);
    if (result == SQLITE_ROW) {
      if (sqlite3_column_type(statement, 0) == SQLITE_NULL) {
        AERROR << "Key-Value database contains a NULL value.";
        sqlite3_finalize(statement);
        return false;
      }
      const auto *text = sqlite3_column_text(statement, 0);
      const int size = sqlite3_column_bytes(statement, 0);
      value->emplace(reinterpret_cast<const char *>(text),
                     static_cast<size_t>(size));
      sqlite3_finalize(statement);
      return true;
    }

    sqlite3_finalize(statement);
    if (result == SQLITE_DONE) {
      return true;
    }
    AERROR << "Failed to read Key-Value database: " << sqlite3_errmsg(db_);
    return false;
  }

 private:
  bool Prepare(const char *sql, sqlite3_stmt **statement) {
    if (db_ == nullptr) {
      AERROR << "Key-Value database is not open.";
      return false;
    }
    if (sqlite3_prepare_v2(db_, sql, -1, statement, nullptr) != SQLITE_OK) {
      AERROR << "Failed to prepare Key-Value database query: "
             << sqlite3_errmsg(db_);
      return false;
    }
    return true;
  }

  bool BindText(sqlite3_stmt *statement, int index, std::string_view value) {
    if (value.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      AERROR << "Key-Value database value is too large.";
      return false;
    }
    const char *text = value.empty() ? "" : value.data();
    if (sqlite3_bind_text(statement, index, text,
                          static_cast<int>(value.size()),
                          SQLITE_TRANSIENT) != SQLITE_OK) {
      AERROR << "Failed to bind Key-Value database query: "
             << sqlite3_errmsg(db_);
      return false;
    }
    return true;
  }

  bool StepDone(sqlite3_stmt *statement) {
    if (sqlite3_step(statement) == SQLITE_DONE) {
      return true;
    }
    AERROR << "Failed to update Key-Value database: " << sqlite3_errmsg(db_);
    return false;
  }

  void Release() {
    if (db_ != nullptr) {
      sqlite3_close(db_);
      db_ = nullptr;
    }
  }

  sqlite3 *db_ = nullptr;
};

}  // namespace

bool KVDB::Put(std::string_view key, std::string_view value) {
  SqliteWraper sqlite;
  return sqlite.Put(key, value);
}

bool KVDB::Delete(std::string_view key) {
  SqliteWraper sqlite;
  return sqlite.Delete(key);
}

bool KVDB::Get(std::string_view key, std::optional<std::string> *value) {
  SqliteWraper sqlite;
  return sqlite.Get(key, value);
}

std::optional<std::string> KVDB::Get(std::string_view key) {
  std::optional<std::string> value;
  if (!Get(key, &value) || !value.has_value() || value->empty()) {
    return {};
  }
  return value;
}

}  // namespace common
}  // namespace apollo
