#include "autan/runtime/run.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <sys/resource.h>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/runtime/diagnostic.hpp"
#include "autan/runtime/round_off.hpp"
#include "comm.hpp"
#include "device.hpp"
#include "run_directory.hpp"

namespace autan::runtime {
namespace {

namespace fs = std::filesystem;

#ifdef AUTAN_HAVE_CUDA
constexpr bool gpu_build = true;
#else
constexpr bool gpu_build = false;
#endif
using Json = nlohmann::ordered_json;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Set by SIGINT and SIGTERM; the loop finalises the run as stopped.
volatile std::sig_atomic_t stop_requested = 0;

// The process's resident memory at its peak so far, in bytes: VmHWM on Linux,
// ru_maxrss elsewhere (bytes on macOS).
std::int64_t host_peak_bytes() {
#if defined(__linux__)
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.starts_with("VmHWM:")) return std::stoll(line.substr(6)) * 1024;  // kB
  }
  return 0;
#else
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<std::int64_t>(usage.ru_maxrss);
#endif
}
extern "C" void request_stop(int /*signal*/) { stop_requested = 1; }

class SignalGuard {
 public:
  SignalGuard() {
    struct sigaction action{};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, &previous_int_);
    sigaction(SIGTERM, &action, &previous_term_);
  }
  SignalGuard(const SignalGuard&) = delete;
  SignalGuard& operator=(const SignalGuard&) = delete;
  ~SignalGuard() {
    sigaction(SIGINT, &previous_int_, nullptr);
    sigaction(SIGTERM, &previous_term_, nullptr);
  }

 private:
  struct sigaction previous_int_{};
  struct sigaction previous_term_{};
};

std::string utc(std::chrono::system_clock::time_point moment, const char* format) {
  const std::time_t t = std::chrono::system_clock::to_time_t(moment);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char text[32];
  std::strftime(text, sizeof text, format, &tm);
  return text;
}

std::string sha256_hex(const std::string& text) {
  std::string hash;
  picosha2::hash256_hex_string(text, hash);
  return hash;
}

std::string read_text(const fs::path& path) {
  const std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw InputError(Diagnostic{.code = "E0705",
                                .message = "cannot read '" + path.string() + "'",
                                .path = "",
                                .data_json = "{}"});
  }
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

[[noreturn]] void invalid(const std::string& path, const std::string& message) {
  throw InputError(Diagnostic{.code = "E0705",
                              .message = path + ": " + message,
                              .path = path,
                              .data_json = nlohmann::json{{"path", path}}.dump()});
}

// results.md: BCType of each resolved condition.
std::string_view bc_type(const BoundaryInput& b) {
  if (b.kind == "wall") return b.temperature ? "BCWallViscousIsothermal" : "BCWallViscousHeatFlux";
  if (b.kind == "slip") return "BCWallInviscid";
  if (b.kind == "symmetry") return "BCSymmetryPlane";
  if (b.kind == "farfield") return "BCFarfield";
  if (b.kind == "outlet") return "BCOutflowSubsonic";
  if (b.kind == "manufactured") return "UserDefined";
  if (b.kind == "oracle-state") return "BCDirichlet";
  return "BCInflowSubsonic";  // inlet, velocity-inlet
}

// Everything the run needs, resolved and checked before the directory exists.
struct Prepared {
  SolverInput input;
  Mesh mesh;
  std::string mesh_hash;
  std::map<std::string, index_t> face_zone;  // boundary zone name -> Mesh::face_zones() index
  std::vector<std::vector<index_t>> monitor_zones;
  const PhysicsFactory* factory = nullptr;
  const PartitionGroups* groups = nullptr;  // the physics' groups; null or empty: none
  std::optional<StartState> start;          // input.start, read and checked
  Json start_averaging;                     // `continue`: the previous run's averaging
  // oracle-state: paired face states per face zone, and the files to archive (sha256, bytes).
  std::map<index_t, std::vector<double>> boundary_states;
  std::map<std::string, std::string> state_files;
  // initial_state: Density, VelocityX/Y/Z and Pressure of every cell, NaN in solid cells.
  std::optional<std::vector<io::Field>> initial;
};

[[noreturn]] void start_mismatch(const std::string& code, const std::string& message,
                                 const std::string& path, const nlohmann::json& data = {}) {
  throw InputError(Diagnostic{.code = code,
                              .message = message,
                              .path = path,
                              .data_json = data.is_null() ? "{}" : data.dump()});
}

// solver-input.md, `periodic`: the pairs of MeshInfo, which must be the
// mesh's.
void check_periodic(const Prepared& p) {
  const auto pairs = p.mesh.periodic_pairs();
  bool same = pairs.size() == p.input.periodic.size();
  for (std::size_t i = 0; same && i < pairs.size(); ++i) {
    const auto& in = p.input.periodic[i];
    same = in.zones[0] == p.mesh.face_zones()[at(pairs[i].zone_a)].name &&
           in.zones[1] == p.mesh.face_zones()[at(pairs[i].zone_b)].name &&
           in.translation == pairs[i].translation;
  }
  if (!same) invalid("periodic", "differs from the periodic pairs of the mesh");
}

// The ghost exchange of a run without ghosts.
class NoGhosts final : public GhostExchange {
 public:
  void forward(std::span<float* const> /*c*/, MemorySpace /*s*/, void* /*stream*/) override {}
  void forward(std::span<double* const> /*c*/, MemorySpace /*s*/, void* /*stream*/) override {}
};

// A private copy of a file and the SHA-256 of the bytes copied: what is
// hashed is what is read, whatever happens to the original meanwhile.
class SnapshotFile {
 public:
  explicit SnapshotFile(const fs::path& from) {
    std::random_device random;
    path_ = fs::temp_directory_path() /
            ("autan-start-" + std::to_string(random()) + "-" + std::to_string(random()) + ".cgns");
    std::ifstream in(from, std::ios::binary);
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    if (!in || !out) throw Error(ErrorCode::io_cannot_open, "cannot read '" + from.string() + "'");
    picosha2::hash256_one_by_one hasher;
    std::vector<char> buffer(1 << 20);
    while (in) {
      in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const auto n = static_cast<std::size_t>(in.gcount());
      hasher.process(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(n));
      out.write(buffer.data(), static_cast<std::streamsize>(n));
    }
    if (in.bad() || !out.flush()) {
      throw Error(ErrorCode::io_cannot_open, "cannot copy '" + from.string() + "'");
    }
    hasher.finish();
    sha256_ = picosha2::get_hash_hex_string(hasher);
  }
  SnapshotFile(const SnapshotFile&) = delete;
  SnapshotFile& operator=(const SnapshotFile&) = delete;
  ~SnapshotFile() {
    std::error_code ec;
    fs::remove(path_, ec);
  }
  [[nodiscard]] const fs::path& path() const { return path_; }
  [[nodiscard]] const std::string& sha256() const { return sha256_; }

 private:
  fs::path path_;
  std::string sha256_;
};

// results.md, run.json `averaging`: null or the four bounds of the interval.
bool is_averaging(const Json& a) {
  if (a.is_null()) return true;
  return a.is_object() && a.size() == 4 && a.contains("from_step") &&
         a["from_step"].is_number_integer() && a.contains("to_step") &&
         a["to_step"].is_number_integer() && a.contains("from_time") &&
         a["from_time"].is_number() && a.contains("to_time") && a["to_time"].is_number();
}

// solver-input.md, `start`: the runtime checks the previous solution's hash,
// mesh, precision and fields, and reads its state, before any directory exists.
void load_start(Prepared& p) {
  const Start& start = *p.input.start;
  const fs::path file = fs::path(start.run) / "solution.cgns";
  const auto unusable = [&](const std::string& why) {
    start_mismatch("E0707", file.string() + ": " + why, "start.run");
  };
  std::optional<SnapshotFile> snapshot;
  try {
    snapshot.emplace(file);
  } catch (const Error& e) {
    unusable(e.what());
  }
  if (snapshot->sha256() != start.solution_sha256) {
    start_mismatch("E0709", "the previous solution's SHA-256 differs from start.solution_sha256",
                   "start.solution_sha256",
                   {{"expected", start.solution_sha256}, {"found", snapshot->sha256()}});
  }
  io::StoredSolution stored;
  try {
    stored = io::read_solution(snapshot->path());
  } catch (const Error& e) {
    unusable(e.what());
  }
  Json metadata;
  try {
    metadata = Json::parse(stored.run_metadata);
  } catch (const Json::exception&) {
    unusable("RunMetadata is not JSON");
  }
  const auto text = [&](const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_string())
      unusable(std::string("RunMetadata has no ") + key);
    return object[key].get<std::string>();
  };
  const std::string mesh =
      text(metadata.contains("mesh") ? metadata["mesh"] : Json::object(), "sha256");
  if (mesh != p.mesh_hash) {
    start_mismatch("E0203", "the previous run was computed on another mesh", "start.run",
                   {{"expected", p.mesh_hash}, {"found", mesh}});
  }
  if (stored.cells != p.mesh.n_cells()) {
    unusable(std::to_string(stored.cells) + " cells, the mesh " + std::to_string(p.mesh.n_cells()));
  }
  const std::string precision = text(metadata, "precision");
  if (precision != p.input.precision) {
    start_mismatch("E0708", "the previous run used precision '" + precision + "'", "start.run");
  }
  std::vector<std::string> names;
  for (const auto& f : stored.fields) names.push_back(f.name);
  if (names != field_names(p.input)) {
    start_mismatch("E0708", "the previous run solved other variables", "start.run");
  }
  StartState state{.fields = std::move(stored.fields), .mean_fields = {}, .restart_fields = {}};
  if (start.mode == "continue") {
    if (!metadata.contains("averaging") || !is_averaging(metadata["averaging"]))
      unusable("RunMetadata has no valid averaging");
    p.start_averaging = metadata["averaging"];
    state.mean_fields = std::move(stored.mean_fields);
    state.restart_fields = std::move(stored.restart_fields);
  }
  p.start = std::move(state);
}

// One zone of an autan.face-states/1 document: rows of centroids (3) and values (5).
struct ZoneStates {
  std::size_t faces = 0;
  std::vector<double> centroids, values;
};

// The zones of a face-state document, or why it is not one (face-states.md, Document): every
// key, every zone and every row are checked, whatever the case uses.
std::map<std::string, ZoneStates> face_state_zones(const nlohmann::json& doc, std::string& why) {
  std::map<std::string, ZoneStates> zones;
  const auto fail = [&](std::string what) {
    why = std::move(what);
    return std::map<std::string, ZoneStates>{};
  };
  if (!doc.is_object()) return fail("not a JSON object");
  for (const auto& [key, value] : doc.items()) {
    if (key != "schema" && key != "oracle" && key != "mesh_sha256" && key != "variables" &&
        key != "zones")
      return fail("unknown key '" + key + "'");
  }
  const auto string_at = [&](const char* key) {
    return doc.contains(key) && doc[key].is_string() ? doc[key].get<std::string>() : std::string();
  };
  if (string_at("schema") != "autan.face-states/1") return fail("schema");
  if (!doc.contains("oracle") || !doc["oracle"].is_string()) return fail("oracle");
  const std::string mesh = string_at("mesh_sha256");
  if (mesh.size() != 64 || mesh.find_first_not_of("0123456789abcdef") != std::string::npos)
    return fail("mesh_sha256");
  if (!doc.contains("variables") ||
      doc["variables"] !=
          nlohmann::json{"density", "velocity-x", "velocity-y", "velocity-z", "pressure"})
    return fail("variables");
  if (!doc.contains("zones") || !doc["zones"].is_object()) return fail("zones");
  for (const auto& [name, z] : doc["zones"].items()) {
    const std::string at_zone = "zones." + name;
    if (!z.is_object() || z.size() != 3 || !z.contains("faces") ||
        !z["faces"].is_number_integer() || z["faces"].get<std::int64_t>() < 0)
      return fail(at_zone + ".faces");
    ZoneStates out;
    out.faces = z["faces"].get<std::size_t>();
    for (const auto& [array, width] : {std::pair{"centroids", 3}, std::pair{"values", 5}}) {
      const auto& rows = z.contains(array) ? z[array] : nlohmann::json();
      if (!rows.is_array() || rows.size() != out.faces) return fail(at_zone + "." + array);
      auto& to = std::string(array) == "centroids" ? out.centroids : out.values;
      for (const auto& row : rows) {
        if (!row.is_array() || row.size() != static_cast<std::size_t>(width))
          return fail(at_zone + "." + array);
        for (const auto& v : row) {
          if (!v.is_number() || !std::isfinite(v.get<double>())) return fail(at_zone + "." + array);
          to.push_back(v.get<double>());
        }
      }
    }
    for (std::size_t r = 0; r < out.faces; ++r) {
      if (!(out.values[5 * r] > 0.0) || !(out.values[5 * r + 4] > 0.0))
        return fail(at_zone + ".values: density and pressure must be > 0");
    }
    zones.emplace(name, std::move(out));
  }
  return zones;
}

// face-states.md: the file of each oracle-state condition, checked by its hash and as a
// document, then against the mesh and the zone, then each face of the zone paired with exactly
// one row by centroid.
void load_states(Prepared& p) {
  std::optional<Geometry> geometry;
  const auto offsets = p.mesh.boundary_zone_offsets();
  for (std::size_t i = 0; i < p.input.boundaries.size(); ++i) {
    const auto& b = p.input.boundaries[i];
    if (!b.states) continue;
    const std::string key = "boundaries[" + std::to_string(i) + "].states";
    const auto fail = [&](const char* code, const std::string& message, const std::string& sub,
                          const nlohmann::json& data = nlohmann::json::object()) {
      throw InputError(Diagnostic{
          .code = code, .message = message, .path = key + sub, .data_json = data.dump()});
    };
    std::error_code ec;
    if (!fs::is_regular_file(b.states->path, ec)) {
      fail("E0420", "'" + b.states->path + "' is not a readable file", ".path");
    }
    std::ifstream in(b.states->path, std::ios::binary);
    if (!in) fail("E0420", "cannot read '" + b.states->path + "'", ".path");
    std::ostringstream text;
    text << in.rdbuf();
    const std::string bytes = text.str();
    const std::string found = sha256_hex(bytes);
    if (found != b.states->sha256) {
      fail("E0420", "the face-state file's SHA-256 differs from states.sha256", ".sha256",
           {{"expected", b.states->sha256}, {"found", found}});
    }
    const nlohmann::json doc = nlohmann::json::parse(bytes, nullptr, false);
    std::string why = doc.is_discarded() ? "not JSON" : "";
    const auto zones =
        why.empty() ? face_state_zones(doc, why) : std::map<std::string, ZoneStates>{};
    if (!why.empty()) {
      fail("E0420", "'" + b.states->path + "' is not an autan.face-states/1 document: " + why,
           ".path", {{"problem", why}});
    }
    if (doc["mesh_sha256"].get<std::string>() != p.mesh_hash) {
      fail("E0422", "the face states were exported for another mesh", ".path",
           {{"expected", p.mesh_hash}, {"found", doc["mesh_sha256"]}});
    }
    const auto it = zones.find(b.zone);
    if (it == zones.end()) {
      fail("E0423", "the face states do not cover zone '" + b.zone + "'", ".path",
           {{"zone", b.zone}});
    }
    const ZoneStates& z = it->second;
    const index_t zone = p.face_zone.at(b.zone);
    const auto first = static_cast<std::size_t>(offsets[at(zone)]);
    const auto n = static_cast<std::size_t>(offsets[at(zone) + 1]) - first;
    if (z.faces != n) {
      fail("E0423", "the face states give zone '" + b.zone + "' another face count", ".path",
           {{"zone", b.zone}, {"expected", n}, {"found", z.faces}});
    }
    if (!geometry) geometry = compute_geometry(p.mesh);
    const auto& g = *geometry;
    // Tolerance: 1e-9 times the largest extent of the bounding box of the zone's nodes.
    std::array<double, 3> lo{}, hi{};
    lo.fill(std::numeric_limits<double>::infinity());
    hi.fill(-std::numeric_limits<double>::infinity());
    const std::array<std::span<const double>, 3> xyz{p.mesh.x(), p.mesh.y(), p.mesh.z()};
    for (std::size_t f = first; f < first + n; ++f) {
      for (const index_t v : p.mesh.face_nodes(static_cast<index_t>(f))) {
        for (std::size_t d = 0; d < 3; ++d) {
          lo[d] = std::min(lo[d], xyz[d][at(v)]);
          hi[d] = std::max(hi[d], xyz[d][at(v)]);
        }
      }
    }
    std::size_t axis = 0;
    for (std::size_t d = 1; d < 3; ++d) {
      if (hi[d] - lo[d] > hi[axis] - lo[axis]) axis = d;
    }
    const double tol = 1e-9 * (hi[axis] - lo[axis]);
    // Rows sorted along the zone's widest axis: a face looks only at the rows within tol on it.
    std::vector<std::size_t> sorted(n);
    for (std::size_t r = 0; r < n; ++r) sorted[r] = r;
    const auto coordinate = [&](std::size_t r) { return z.centroids[3 * r + axis]; };
    std::ranges::sort(sorted, {}, coordinate);
    std::vector<char> used(n, 0);
    std::vector<double> paired(5 * n);
    for (std::size_t j = 0; j < n; ++j) {
      const std::size_t f = first + j;
      const std::array<double, 3> c{g.face_centroid_x[f], g.face_centroid_y[f],
                                    g.face_centroid_z[f]};
      std::size_t match = n, count = 0;
      for (auto r = std::ranges::lower_bound(sorted, c[axis] - tol, {}, coordinate);
           r != sorted.end() && coordinate(*r) <= c[axis] + tol; ++r) {
        const double* q = &z.centroids[3 * *r];
        if (std::hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) <= tol) {
          match = *r;
          ++count;
        }
      }
      if (count != 1 || used[match] != 0) {
        fail("E0421",
             "face " + std::to_string(j) + " of zone '" + b.zone + "' matches " +
                 std::to_string(count) + " rows of the face states",
             ".path", {{"zone", b.zone}, {"face", j}, {"centroid", c}, {"matches", count}});
      }
      used[match] = 1;
      for (std::size_t k = 0; k < 5; ++k) paired[k * n + j] = z.values[5 * match + k];
    }
    p.boundary_states[zone] = std::move(paired);
    p.state_files.emplace(found, bytes);
  }
}

// initial-state.md: the file read once and checked (E0508), against the mesh (E0509) and its
// fluid cells (E0510), then every fluid cell paired with exactly one row by centroid, and every
// row with exactly one cell (E0511), on the whole mesh.
void load_initial(Prepared& p) {
  const InitialStateFile& file = *p.input.initial_state;
  const auto fail = [&](const char* code, const std::string& message,
                        const nlohmann::json& data = nlohmann::json::object()) {
    throw InputError(Diagnostic{
        .code = code, .message = message, .path = "initial_state.path", .data_json = data.dump()});
  };
  std::error_code ec;
  if (!fs::is_regular_file(file.path, ec)) fail("E0508", "'" + file.path + "' is not a file");
  std::ifstream in(file.path, std::ios::binary);
  if (!in) fail("E0508", "cannot read '" + file.path + "'");
  std::ostringstream text;
  text << in.rdbuf();
  const std::string bytes = text.str();
  const std::string found = sha256_hex(bytes);
  if (found != file.sha256) {
    fail("E0508", "the initial-state file's SHA-256 differs from initial_state.sha256",
         {{"expected", file.sha256}, {"found", found}});
  }
  const nlohmann::json doc = nlohmann::json::parse(bytes, nullptr, false);
  const auto invalid_document = [&](const std::string& why) {
    fail("E0508", "'" + file.path + "' is not an autan.initial-state/1 document: " + why,
         {{"problem", why}});
  };
  if (doc.is_discarded() || !doc.is_object()) invalid_document("not a JSON object");
  for (const auto& [key, value] : doc.items()) {
    if (key != "schema" && key != "oracle" && key != "mesh_sha256" && key != "variables" &&
        key != "cells" && key != "centroids" && key != "values")
      invalid_document("unknown key '" + key + "'");
  }
  if (!doc.contains("schema") || doc["schema"] != "autan.initial-state/1")
    invalid_document("schema");
  if (!doc.contains("oracle") || !doc["oracle"].is_string()) invalid_document("oracle");
  if (!doc.contains("mesh_sha256") || !doc["mesh_sha256"].is_string() ||
      doc["mesh_sha256"].get<std::string>().size() != 64 ||
      doc["mesh_sha256"].get<std::string>().find_first_not_of("0123456789abcdef") !=
          std::string::npos)
    invalid_document("mesh_sha256");
  if (!doc.contains("variables") ||
      doc["variables"] !=
          nlohmann::json{"density", "velocity-x", "velocity-y", "velocity-z", "pressure"})
    invalid_document("variables");
  if (!doc.contains("cells") || !doc["cells"].is_number_integer() ||
      doc["cells"].get<std::int64_t>() < 1)
    invalid_document("cells");
  const auto rows = doc["cells"].get<std::size_t>();
  std::vector<double> centroids, values;
  for (const auto& [array, width] : {std::pair{"centroids", 3}, std::pair{"values", 5}}) {
    const auto& a = doc.contains(array) ? doc[array] : nlohmann::json();
    if (!a.is_array() || a.size() != rows) invalid_document(array);
    auto& to = std::string(array) == "centroids" ? centroids : values;
    to.reserve(rows * static_cast<std::size_t>(width));
    for (const auto& row : a) {
      if (!row.is_array() || row.size() != static_cast<std::size_t>(width)) invalid_document(array);
      for (const auto& v : row) {
        if (!v.is_number() || !std::isfinite(v.get<double>())) invalid_document(array);
        to.push_back(v.get<double>());
      }
    }
  }
  for (std::size_t r = 0; r < rows; ++r) {
    if (!(values[5 * r] > 0.0) || !(values[5 * r + 4] > 0.0))
      invalid_document("values: density and pressure must be > 0");
  }
  if (doc["mesh_sha256"].get<std::string>() != p.mesh_hash) {
    fail("E0509", "the initial state was written for another mesh",
         {{"expected", p.mesh_hash}, {"found", doc["mesh_sha256"]}});
  }
  // Fluid cells: those of zones whose initial state is a fluid's.
  std::map<std::string, bool> fluid_zone;
  for (const auto& z : p.input.cell_zones) fluid_zone[z.zone] = z.initial.kind == "fluid";
  const Mesh& mesh = p.mesh;
  std::vector<index_t> fluid;
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    if (fluid_zone[mesh.cell_zones()[at(mesh.cell_zone()[at(c)])].name]) fluid.push_back(c);
  }
  if (rows != fluid.size()) {
    fail("E0510",
         "the initial state has " + std::to_string(rows) + " rows for " +
             std::to_string(fluid.size()) + " fluid cells",
         {{"expected", fluid.size()}, {"found", rows}});
  }
  // Tolerance: 1e-9 times the largest extent of the mesh's bounding box.
  const std::array<std::span<const double>, 3> xyz{mesh.x(), mesh.y(), mesh.z()};
  double extent = 0.0;
  for (const auto& coordinate : xyz) {
    const auto [lo, hi] = std::ranges::minmax_element(coordinate);
    extent = std::max(extent, *hi - *lo);
  }
  const double tol = 1e-9 * extent;
  const Geometry g = compute_geometry(mesh);
  const auto centroid = [&](index_t c) {
    const auto i = at(c);
    return std::array<double, 3>{g.cell_centroid_x[i], g.cell_centroid_y[i], g.cell_centroid_z[i]};
  };
  // Points binned in cubes of side tol: two points within tol lie in
  // neighbouring bins, whatever their alignment, so pairing is linear.
  using Bin = std::array<std::int64_t, 3>;
  struct BinHash {
    std::size_t operator()(const Bin& b) const noexcept {
      return std::hash<std::int64_t>{}(b[0] * 73856093 ^ b[1] * 19349663 ^ b[2] * 83492791);
    }
  };
  const auto bin = [&](const std::array<double, 3>& x) {
    return Bin{static_cast<std::int64_t>(std::floor(x[0] / tol)),
               static_cast<std::int64_t>(std::floor(x[1] / tol)),
               static_cast<std::int64_t>(std::floor(x[2] / tol))};
  };
  const auto near = [&](const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
  };
  // The 27 bins around b, each visited with `visit`.
  const auto around = [](const Bin& b, const auto& visit) {
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      for (std::int64_t dy = -1; dy <= 1; ++dy) {
        for (std::int64_t dz = -1; dz <= 1; ++dz) visit(Bin{b[0] + dx, b[1] + dy, b[2] + dz});
      }
    }
  };
  // Within tol of each other, two fluid cells cannot be told apart.
  std::unordered_map<Bin, std::vector<index_t>, BinHash> cell_bins;
  for (const index_t c : fluid) cell_bins[bin(centroid(c))].push_back(c);
  for (const index_t c : fluid) {
    around(bin(centroid(c)), [&](const Bin& b) {
      const auto it = cell_bins.find(b);
      if (it == cell_bins.end()) return;
      for (const index_t other : it->second) {
        if (other > c && near(centroid(c), centroid(other))) {
          fail("E0511",
               "cells " + std::to_string(c) + " and " + std::to_string(other) +
                   " lie within the pairing tolerance of each other",
               {{"cells", {c, other}}, {"tolerance", tol}});
        }
      }
    });
  }
  std::unordered_map<Bin, std::vector<std::size_t>, BinHash> row_bins;
  for (std::size_t r = 0; r < rows; ++r)
    row_bins[bin({centroids[3 * r], centroids[3 * r + 1], centroids[3 * r + 2]})].push_back(r);
  std::vector<std::int64_t> owner(rows, -1);  // the cell each row is paired with
  constexpr double unset = std::numeric_limits<double>::quiet_NaN();
  std::vector<io::Field> fields{{.name = "Density", .values = {}},
                                {.name = "VelocityX", .values = {}},
                                {.name = "VelocityY", .values = {}},
                                {.name = "VelocityZ", .values = {}},
                                {.name = "Pressure", .values = {}}};
  for (auto& f : fields) f.values.assign(at(mesh.n_cells()), unset);
  for (const index_t c : fluid) {
    const auto x = centroid(c);
    std::size_t match = rows, count = 0;
    around(bin(x), [&](const Bin& b) {
      const auto it = row_bins.find(b);
      if (it == row_bins.end()) return;
      for (const std::size_t r : it->second) {
        if (near(x, {centroids[3 * r], centroids[3 * r + 1], centroids[3 * r + 2]})) {
          match = r;
          ++count;
        }
      }
    });
    if (count != 1) {
      fail("E0511",
           "cell " + std::to_string(c) + " matches " + std::to_string(count) +
               " rows of the initial state",
           {{"cells", {c}}, {"centroid", x}, {"matches", count}});
    }
    if (owner[match] >= 0) {
      fail("E0511",
           "cells " + std::to_string(owner[match]) + " and " + std::to_string(c) +
               " match one row of the initial state",
           {{"cells", {owner[match], c}}, {"centroid", x}});
    }
    owner[match] = c;
    for (std::size_t k = 0; k < 5; ++k) fields[k].values[at(c)] = values[5 * match + k];
  }
  p.initial = std::move(fields);
  p.state_files.emplace(found, bytes);
}

Prepared prepare(const fs::path& input_file, const PhysicsRegistry& registry, int ranks) {
  Prepared p{.input = parse_solver_input(read_text(input_file)),
             .mesh = {},
             .mesh_hash = {},
             .face_zone = {},
             .monitor_zones = {},
             .factory = nullptr,
             .start = std::nullopt,
             .start_averaging = {},
             .boundary_states = {},
             .state_files = {},
             .initial = std::nullopt};
  try {
    p.mesh = io::read_mesh(p.input.mesh_path);
    p.mesh_hash = io::mesh_hash(p.input.mesh_path);
  } catch (const Error& e) {
    throw InputError(Diagnostic{.code = error_code_string(e.code()),
                                .message = e.what(),
                                .path = "mesh.path",
                                .data_json = "{}"});
  }
  if (p.mesh_hash != p.input.mesh_sha256) {
    throw InputError(Diagnostic{
        .code = "E0203",
        .message = "the mesh file's SHA-256 differs from mesh.sha256 of the input",
        .path = "mesh.sha256",
        .data_json =
            nlohmann::json{{"expected", p.input.mesh_sha256}, {"found", p.mesh_hash}}.dump()});
  }
  check_periodic(p);
  // solver-input.md, `devices`: one rank per device, as many as the launcher started.
  if (p.input.device_count != ranks) {
    invalid("devices.count", "is " + std::to_string(p.input.device_count) + ", but " +
                                 std::to_string(ranks) + " ranks run the case");
  }
  if (p.input.start) load_start(p);
  if (p.input.verification) {
    // The manufactured solution is checked like the mesh: by its hash.
    auto& v = *p.input.verification;
    const std::ifstream in(v.path, std::ios::binary);
    if (!in) {
      throw InputError(Diagnostic{.code = "E0207",
                                  .message = "cannot read '" + v.path + "'",
                                  .path = "verification.path",
                                  .data_json = "{}"});
    }
    std::ostringstream text;
    text << in.rdbuf();
    v.text = text.str();
    if (sha256_hex(v.text) != v.sha256) {
      throw InputError(Diagnostic{
          .code = "E0206",
          .message = "the manufactured solution's SHA-256 differs from verification.sha256",
          .path = "verification.sha256",
          .data_json =
              nlohmann::json{{"expected", v.sha256}, {"found", sha256_hex(v.text)}}.dump()});
    }
  }
  const io::MeshInfoZones zones = io::meshinfo_zones(p.mesh);
  for (const index_t z : zones.boundary) p.face_zone[p.mesh.face_zones()[at(z)].name] = z;
  std::map<std::string, bool> conditioned;
  for (std::size_t i = 0; i < p.input.boundaries.size(); ++i) {
    const auto& name = p.input.boundaries[i].zone;
    if (!p.face_zone.contains(name))
      invalid("boundaries[" + std::to_string(i) + "].zone", "no boundary zone '" + name + "'");
    if (conditioned[name])
      invalid("boundaries[" + std::to_string(i) + "].zone",
              "zone '" + name + "' has two conditions");
    // A periodic zone joins its partner; it takes no condition.
    if (p.mesh.face_zones()[at(p.face_zone.at(name))].kind == BcKind::periodic)
      invalid("boundaries[" + std::to_string(i) + "].zone", "zone '" + name + "' is periodic");
    conditioned[name] = true;
  }
  const auto offsets = p.mesh.boundary_zone_offsets();
  for (const auto& [name, z] : p.face_zone) {
    if (offsets[at(z) + 1] > offsets[at(z)] && !conditioned[name] &&
        p.mesh.face_zones()[at(z)].kind != BcKind::periodic) {
      invalid("boundaries", "boundary zone '" + name + "' has faces and no condition");
    }
  }
  std::map<std::string, bool> described;
  std::map<std::string, bool> in_mesh;
  for (const index_t z : zones.cell) in_mesh[p.mesh.cell_zones()[at(z)].name] = true;
  for (std::size_t i = 0; i < p.input.cell_zones.size(); ++i) {
    const auto& name = p.input.cell_zones[i].zone;
    const std::string path = "cell_zones[" + std::to_string(i) + "].zone";
    if (!in_mesh[name]) invalid(path, "no cell zone '" + name + "' holds cells in the mesh");
    if (described[name]) invalid(path, "cell zone '" + name + "' is described twice");
    described[name] = true;
  }
  for (const index_t z : zones.cell) {
    if (!described[p.mesh.cell_zones()[at(z)].name]) {
      invalid("cell_zones", "cell zone '" + p.mesh.cell_zones()[at(z)].name + "' has no entry");
    }
  }
  for (std::size_t m = 0; m < p.input.monitors.size(); ++m) {
    std::vector<index_t> ids;
    for (const auto& name : p.input.monitors[m].zones) {
      if (!p.face_zone.contains(name))
        invalid("monitors[" + std::to_string(m) + "].zones", "no boundary zone '" + name + "'");
      // A periodic zone is no surface the fluid exerts a load on.
      if (p.mesh.face_zones()[at(p.face_zone.at(name))].kind == BcKind::periodic)
        invalid("monitors[" + std::to_string(m) + "].zones", "zone '" + name + "' is periodic");
      ids.push_back(p.face_zone.at(name));
    }
    p.monitor_zones.push_back(std::move(ids));
  }
  // A point vortex takes its circulation from the lift on walls (slip walls in inviscid flow).
  std::map<std::string, std::string> condition;
  for (const auto& b : p.input.boundaries) condition[b.zone] = b.kind;
  for (std::size_t i = 0; i < p.input.boundaries.size(); ++i) {
    const auto& vortex = p.input.boundaries[i].point_vortex;
    if (!vortex) continue;
    const std::string path = "boundaries[" + std::to_string(i) + "].point_vortex.zones";
    for (const auto& name : vortex->zones) {
      if (!p.face_zone.contains(name)) invalid(path, "no boundary zone '" + name + "'");
      const auto it = condition.find(name);
      if (it == condition.end() || (it->second != "wall" && it->second != "slip"))
        invalid(path, "zone '" + name + "' is not a wall");
    }
  }
  load_states(p);
  if (p.input.initial_state) load_initial(p);
  // A device run needs a build that drives that device and a device to
  // drive; neither falls back to the CPU.
  if (p.input.device_kind != "cpu") {
    const bool built = p.input.device_kind == "cuda" && gpu_build;
    if (!built) {
      throw InputError(Diagnostic{
          .code = "E0706",
          .message = "this build of autan-run does not drive " + p.input.device_kind + " devices",
          .path = "devices.kind",
          .data_json = nlohmann::json{{"path", "devices.kind"}}.dump()});
    }
    if (visible_gpus() == 0) {
      throw InputError(Diagnostic{
          .code = "E0807",
          .message = "devices.kind is cuda, but this process sees no CUDA device",
          .path = "devices.kind",
          .data_json =
              nlohmann::json{{"visible", 0}, {"needed", 1}, {"path", "devices.kind"}}.dump()});
    }
  }
  const auto* entry = registry.find(p.input);
  p.factory = entry != nullptr ? &entry->factory : nullptr;
  p.groups = entry != nullptr ? &entry->groups : nullptr;
  if (entry != nullptr && entry->scope) {
    if (auto refusal = entry->scope(p.input, ranks)) {
      auto data = nlohmann::json::parse(refusal->data_json);
      data["path"] = refusal->path;
      throw InputError(Diagnostic{.code = "E0706",
                                  .message = std::move(refusal->message),
                                  .path = std::move(refusal->path),
                                  .data_json = data.dump()});
    }
  }
  if (entry != nullptr && ranks > 1 && !entry->several_ranks) {
    throw InputError(Diagnostic{
        .code = "E0706",
        .message =
            "this build's physics for the input runs on one rank, not " + std::to_string(ranks),
        .path = "physics",
        .data_json = nlohmann::json{{"path", "physics"}, {"ranks", ranks}}.dump()});
  }
  if (p.factory == nullptr) {
    const auto& in = p.input;
    throw InputError(Diagnostic{.code = "E0706",
                                .message = "this build of autan-run has no physics for " +
                                           in.physics_kind + " flow, turbulence '" +
                                           in.turbulence.model + "', precision " + in.precision +
                                           " on " + in.device_kind,
                                .path = "physics",
                                .data_json = nlohmann::json{{"kind", in.physics_kind},
                                                            {"turbulence", in.turbulence.model},
                                                            {"precision", in.precision},
                                                            {"device", in.device_kind}}
                                                 .dump()});
  }
  return p;
}

// Coefficients of a forces monitor (results.md, History).
Json forces_record(const Loads& loads, const ForcesMonitor& monitor, const Freestream& fs) {
  Json r{{"fx", loads.force[0]},  {"fy", loads.force[1]},  {"fz", loads.force[2]},
         {"mx", loads.moment[0]}, {"my", loads.moment[1]}, {"mz", loads.moment[2]}};
  if (!monitor.ref_area) return r;
  const double q = 0.5 * fs.density * fs.speed * fs.speed;
  const double a = fs.aoa, b = fs.sideslip;
  const std::array<double, 3> drag{std::cos(a) * std::cos(b), -std::sin(b),
                                   std::sin(a) * std::cos(b)};
  const std::array<double, 3> lift{-std::sin(a), 0.0, std::cos(a)};
  const std::array<double, 3> side{lift[1] * drag[2] - lift[2] * drag[1],
                                   lift[2] * drag[0] - lift[0] * drag[2],
                                   lift[0] * drag[1] - lift[1] * drag[0]};
  const auto dot = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const double qa = q * *monitor.ref_area;
  r["cd"] = dot(loads.force, drag) / qa;
  r["cs"] = dot(loads.force, side) / qa;
  r["cl"] = dot(loads.force, lift) / qa;
  if (monitor.ref_length) {
    const double qal = qa * *monitor.ref_length;
    r["cmx"] = loads.moment[0] / qal;
    r["cmy"] = loads.moment[1] / qal;
    r["cmz"] = loads.moment[2] / qal;
  }
  return r;
}

class Run final : public Log {
 public:
  Run(Prepared& p, const SolverBuild& build, Comm& comm,
      std::chrono::steady_clock::time_point launched)
      : p_(p),
        comm_(comm),
        launched_(launched),
        geometry_(compute_geometry(p.mesh)),
        groups_(p.groups != nullptr && *p.groups ? (*p.groups)(p.input, p.mesh, geometry_)
                                                 : CellGroups{}),
        local_(make_part()) {
    if (p_.input.device_kind == "cuda") gpu_.emplace(comm_.local_rank());
    if (local_) {
      ghosts_ = comm_.ghost_exchange(*local_);
      if (p_.start) local_start_ = owned_start(*p_.start);
      if (p_.initial) local_initial_ = owned_fields(*p_.initial);
    } else {
      ghosts_ = std::make_unique<NoGhosts>();
    }
    colour_cells_of_physics();
    // Rank 0 names the run: its clock and its random suffix.
    const auto started = std::chrono::system_clock::now();
    std::random_device random;
    char suffix[9];
    std::snprintf(suffix, sizeof suffix, "%08x", static_cast<unsigned>(random()));
    const std::string run_id =
        comm_.broadcast("run-" + utc(started, "%Y%m%dT%H%M%SZ") + "-" + suffix);
    const std::string started_at = comm_.broadcast(utc(started, "%Y-%m-%dT%H:%M:%SZ"));
    const auto& in = p_.input;
    const std::string backend = in.device_kind;
    std::string parts = in.case_sha256 + "\n" + p_.mesh_hash + "\n" + build.commit + "\n" +
                        in.precision + "\n" + backend + "\n" + in.device_kind + "\n" +
                        std::to_string(in.device_count);
    Json start = nullptr;
    if (in.start) {
      const Start& s = *in.start;
      parts += "\n" + s.identity + "\n" + s.mode + "\n" + std::to_string(s.iterations) + "\n" +
               s.solution_sha256;
      Json reference = nullptr;
      if (s.residual_reference) {
        const auto equations = residual_names(in);
        reference = Json::object();
        for (std::size_t e = 0; e < equations.size(); ++e)
          reference[equations[e]] = (*s.residual_reference)[e];
      }
      start = Json{{"run_id", s.run_id},
                   {"identity", s.identity},
                   {"mode", s.mode},
                   {"iterations", s.iterations},
                   {"time", s.time ? Json(*s.time) : Json(nullptr)},
                   {"solution_sha256", s.solution_sha256},
                   {"residual_reference", reference},
                   {"woken", s.woken ? Json(*s.woken) : Json(nullptr)}};
    }
    const std::string identity = sha256_hex(parts);
    info_ = Json{
        {"schema", "autan.results/1"},
        {"run_id", run_id},
        {"identity", identity},
        {"status", "running"},
        {"case", {{"name", in.case_name}, {"sha256", in.case_sha256}}},
        {"mesh",
         {{"path", fs::absolute(in.mesh_path).lexically_normal().string()},
          {"sha256", p_.mesh_hash}}},
        {"solver", {{"version", build.version}, {"commit", build.commit}, {"backend", backend}}},
        {"precision", in.precision},
        {"deterministic", in.deterministic},
        {"devices", {{"kind", in.device_kind}, {"count", in.device_count}}},
        {"started", started_at},
        {"finished", nullptr},
        {"iterations", 0},
        {"time", in.time.kind == "unsteady" ? Json(0.0) : Json(nullptr)},
        {"averaging", nullptr},
        {"convergence", nullptr},
        {"start", start},
        {"monitors", nullptr},
        {"timing", nullptr},
        {"diagnostics", Json::array()}};
    if (continued()) {
      info_["iterations"] = in.start->iterations;
      if (in.time.kind == "unsteady") info_["time"] = *in.start->time;
      info_["averaging"] = p_.start_averaging;
    }
  }

  // Rank 0 creates the directory; the others append to it once it exists.
  void open(const fs::path& runs) {
    std::string path;
    if (comm_.rank() == 0) {
      try {
        dir_ = std::make_unique<RunDirectory>(
            runs, info_["run_id"].get<std::string>(), info_.dump(2), p_.input.case_toml,
            io::mesh_info_json(p_.mesh, p_.mesh_hash),
            log_line("info", "solver", "run " + info_["run_id"].get<std::string>() + " started", "",
                     Json{{"cells", p_.mesh.n_cells()},
                          {"faces", p_.mesh.n_faces()},
                          {"ranks", comm_.size()}}
                         .dump()),
            p_.input.verification ? std::optional<std::string>(p_.input.verification->text)
                                  : std::nullopt,
            p_.state_files);
        path = dir_->path().string();
      } catch (...) {
        (void)comm_.broadcast({});
        throw;
      }
    }
    path = comm_.broadcast(path);
    if (path.empty()) throw std::runtime_error("rank 0 could not create the run directory");
    if (comm_.rank() != 0) dir_ = std::make_unique<RunDirectory>(fs::path(path));
    if (comm_.size() > 1) log_partition();
    if (gpu_) {
      note(
          "info", "cuda",
          "rank 0 runs on CUDA device " + std::to_string(gpu_->index()) + " (" + gpu_->name() + ")",
          "",
          Json{{"device", gpu_->index()}, {"name", gpu_->name()}, {"visible", visible_gpus()}}
              .dump());
    }
  }
  [[nodiscard]] const fs::path& path() const { return dir_->path(); }
  void stop_on(fs::path file) { stop_file_ = std::move(file); }

  ExitStatus execute() {
    const StartState* start = nullptr;
    if (p_.start) start = local_start_ ? &*local_start_ : &*p_.start;
    const std::vector<io::Field>* initial = nullptr;
    if (p_.initial) initial = local_initial_ ? &*local_initial_ : &*p_.initial;
    const PhysicsContext context{.mesh = physics_mesh(),
                                 .geometry = physics_geometry(),
                                 .input = p_.input,
                                 .log = *this,
                                 .start = start,
                                 .ghosts = *ghosts_,
                                 .ranks = comm_,
                                 .boundary_states = p_.boundary_states,
                                 .initial = initial,
                                 .device = gpu_ ? gpu_->index() : -1,
                                 .stream = gpu_ ? gpu_->stream() : nullptr,
                                 .cell_colours = colours_,
                                 .n_colours = n_colours_,
                                 .group_offsets = group_offsets_,
                                 .group_cells = group_cells_,
                                 .cell_group_colours = group_colours_,
                                 .n_group_colours = n_group_colours_};
    std::string status = "running";
    std::int64_t done = continued() ? p_.input.start->iterations : 0;
    try {
      physics_ = (*p_.factory)(context);
      host_after_setup_ = host_peak_bytes();
      if (gpu_) device_after_setup_ = gpu_->sample();
      set_floor();
      status = std::string(iterate(done));
    } catch (const std::exception& e) {
      return fail(e.what(), done);
    }
    return finish(status, done);
  }

 private:
  // case-schema.md#time: in fp32+acc64, a steady run counts an equation
  // converged at floor_k = round_off_floor * Phi_k at the latest, Phi_k the
  // physics' constant flux scale; run.json records the floors.
  void set_floor() {
    const auto& in = p_.input;
    const auto equations = residual_names(in);
    const std::vector<double> scales = physics_->residual_scales();
    if (scales.empty()) return;
    if (scales.size() != equations.size()) {
      throw std::runtime_error("the physics gave " + std::to_string(scales.size()) +
                               " residual scales for " + std::to_string(equations.size()) +
                               " equations");
    }
    for (const double s : scales) {
      if (!std::isfinite(s) || s < 0.0)
        throw std::runtime_error("the physics gave a residual scale that is not finite and >= 0");
    }
    if (in.time.kind != "steady" || in.precision != "fp32+acc64") return;
    Json floors = Json::object();
    for (std::size_t e = 0; e < equations.size(); ++e) {
      floor_.push_back(round_off_floor * scales[e]);
      floors[equations[e]] = floor_.back();
    }
    info_["convergence"] = Json{{"floor", floors}};
    if (comm_.rank() == 0) dir_->write_run_json(info_.dump(2));
  }

  // On several ranks, each runs its part of the mesh, partitioned by METIS
  // with the face-count cost of each cell and the physics' groups kept
  // whole; every rank partitions the same
  // mesh with the same seed, so all agree on the parts (partitioning.md).
  // On one rank, a mesh with periodic pairs runs on its one-part partition,
  // whose image ghosts make each periodic face an interior face; its owned
  // cells are the mesh's cells, in order. Otherwise, the mesh itself.
  std::optional<LocalPartition> make_part() {
    const Mesh& mesh = p_.mesh;
    const std::int32_t n = comm_.size();
    if (n == 1 && mesh.periodic_pairs().empty()) return std::nullopt;
    std::vector<std::int32_t> parts(at(mesh.n_cells()), 0);
    if (n > 1) {
      parts = partition_cells(mesh, n,
                              {.cell_weights = cell_cost_weights(mesh),
                               .group_offsets = groups_.offsets,
                               .group_cells = groups_.cells});
    }
    LocalPartition part = extract_partition(mesh, geometry_, parts, n, comm_.rank());
    comm_.fill_ghosts(part);
    return part;
  }

  // This rank's owned cells' share of cell fields of the mesh.
  [[nodiscard]] std::vector<io::Field> owned_fields(const std::vector<io::Field>& fields) const {
    const auto ids = owned_ids();
    std::vector<io::Field> out;
    for (const auto& f : fields) {
      io::Field part{.name = f.name, .values = {}};
      part.values.reserve(ids.size());
      for (const auto id : ids) part.values.push_back(f.values.at(static_cast<std::size_t>(id)));
      out.push_back(std::move(part));
    }
    return out;
  }

  // The start state of this rank's owned cells, from that of the mesh.
  [[nodiscard]] StartState owned_start(const StartState& global) const {
    return StartState{.fields = owned_fields(global.fields),
                      .mean_fields = owned_fields(global.mean_fields),
                      .restart_fields = owned_fields(global.restart_fields)};
  }

  // The mesh index of each owned cell of the physics' mesh.
  [[nodiscard]] std::vector<std::int64_t> owned_ids() const {
    const Mesh& mesh = physics_mesh();
    const auto global = mesh.cell_global_id();
    std::vector<std::int64_t> ids(at(mesh.n_owned_cells()));
    for (std::size_t c = 0; c < ids.size(); ++c) ids[c] = local_ ? global[c] : std::int64_t(c);
    return ids;
  }

  // Owned and ghost cells of each rank, in log.jsonl.
  void log_partition() {
    const Mesh& mesh = physics_mesh();
    std::vector<ExactSum> counts(2 * at(comm_.size()));
    counts[2 * at(comm_.rank())].add(static_cast<double>(mesh.n_owned_cells()));
    counts[2 * at(comm_.rank()) + 1].add(
        static_cast<double>(mesh.n_cells() - mesh.n_owned_cells()));
    comm_.sum(counts);
    Json owned = Json::array(), ghosts = Json::array();
    for (std::size_t r = 0; r < at(comm_.size()); ++r) {
      owned.push_back(static_cast<std::int64_t>(counts[2 * r].value()));
      ghosts.push_back(static_cast<std::int64_t>(counts[2 * r + 1].value()));
    }
    write("info", "mpi", "mesh partitioned over " + std::to_string(comm_.size()) + " ranks", "",
          Json{{"ranks", comm_.size()}, {"owned_cells", owned}, {"ghost_cells", ghosts}}.dump());
  }
  // The global colouring of the mesh (colour_cells), for each cell of the
  // physics' mesh through its global id: every rank computes the same.
  // With the physics' groups, their colouring and the groups this rank owns
  // (by increasing least global id, each in the physics' order) likewise.
  void colour_cells_of_physics() {
    const Mesh& mesh = p_.mesh;
    const auto z = [](std::int64_t i) { return static_cast<std::size_t>(i); };
    const auto global_ids = mesh.cell_global_id();
    const auto ids = physics_mesh().cell_global_id();
    const auto of_physics = [&](const std::vector<std::int32_t>& colours,
                                std::vector<std::int32_t>& out, std::int32_t& n_out) {
      std::vector<std::int32_t> by_id(colours.size());
      for (std::size_t c = 0; c < colours.size(); ++c) by_id[z(global_ids[c])] = colours[c];
      out.resize(ids.size());
      for (std::size_t c = 0; c < ids.size(); ++c) out[c] = by_id[z(ids[c])];
      for (const std::int32_t k : colours) n_out = std::max(n_out, k + 1);
    };
    of_physics(colour_cells(mesh), colours_, n_colours_);
    if (p_.groups == nullptr || !*p_.groups) return;
    of_physics(colour_cells(mesh, groups_.offsets, groups_.cells), group_colours_,
               n_group_colours_);
    std::vector<index_t> owned_at(at(mesh.n_cells()), -1);  // by global id
    for (index_t c = 0; c < physics_mesh().n_owned_cells(); ++c) owned_at[z(ids[at(c)])] = c;
    std::vector<std::pair<std::int64_t, std::size_t>> order;  // (least global id, group)
    for (std::size_t g = 0; g + 1 < groups_.offsets.size(); ++g) {
      std::int64_t least = std::numeric_limits<std::int64_t>::max();
      for (auto k = groups_.offsets[g]; k < groups_.offsets[g + 1]; ++k)
        least = std::min(least, global_ids[at(groups_.cells[z(k)])]);
      order.emplace_back(least, g);
    }
    std::ranges::sort(order);
    group_offsets_.assign(1, 0);
    for (const auto& [least, g] : order) {
      if (owned_at[z(least)] < 0) continue;
      for (auto k = groups_.offsets[g]; k < groups_.offsets[g + 1]; ++k)
        group_cells_.push_back(owned_at[z(global_ids[at(groups_.cells[z(k)])])]);
      group_offsets_.push_back(static_cast<offset_t>(group_cells_.size()));
    }
  }

  [[nodiscard]] const Mesh& physics_mesh() const { return local_ ? local_->mesh : p_.mesh; }
  [[nodiscard]] const Geometry& physics_geometry() const {
    return local_ ? local_->geometry : geometry_;
  }

  [[nodiscard]] bool continued() const {
    return p_.input.start && p_.input.start->mode == "continue";
  }

  // Thrown by the solution writer when a physics member fails: unlike a
  // write failure, it fails even a diverged run (runtime.md).
  struct PhysicsFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  void record_failure(const std::string& message) {
    add_diagnostic(Diagnostic{
        .code = "E0704",
        .message = message,
        .path = "",
        .data_json = comm_.size() > 1 ? Json{{"rank", comm_.rank()}}.dump() : std::string("{}")});
  }

  // A run diagnostic goes to run.json and, with the same code and data, to
  // log.jsonl (results.md). Every rank records it; rank 0 logs it.
  void add_diagnostic(const Diagnostic& d) {
    const Json entry = Json::parse(d.json());
    info_["diagnostics"].push_back(entry);
    // The level follows the code's letter (results.md, Log).
    note(d.code.starts_with('W') ? "warning" : "error", "solver", d.message, d.code,
         entry["data"].dump());
  }

  // An event of the run, which every rank knows: rank 0 logs it.
  void note(std::string_view level, std::string_view source, std::string_view message,
            std::string_view code, std::string_view data_json) {
    if (comm_.rank() == 0) dir_->append_log(log_line(level, source, message, code, data_json));
  }

  // On several ranks, a physics failure on one rank leaves the others
  // waiting in a collective call: that rank records E0704 and the failed
  // run.json, and ends every process with exit status 1.
  ExitStatus fail(const std::string& message, std::int64_t done) {
    if (comm_.size() == 1) {
      record_failure(message);
      return finish("failed", done);
    }
    // The abort follows whatever these writes meet.
    try {
      record_failure(message);
      if (comm_.rank() != 0) {
        const Json& entry = info_["diagnostics"].back();
        dir_->append_log(log_line("error", "solver", message, "E0704", entry["data"].dump()));
      }
      info_["iterations"] = done;
      info_["status"] = "failed";
      info_["finished"] = utc(std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%SZ");
      dir_->write_run_json(info_.dump(2));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n",
                   Diagnostic{.code = "E0704",
                              .message = message + "; recording it: " + e.what(),
                              .path = "",
                              .data_json = Json{{"rank", comm_.rank()}}.dump()}
                       .json()
                       .c_str());
    }
    comm_.abort(static_cast<int>(ExitStatus::failed));
    return ExitStatus::failed;
  }

 public:
  // A physics' event: rank 0 logs all, the other ranks their warnings and
  // errors, with their rank in `data`.
  void write(std::string_view level, std::string_view source, std::string_view message,
             std::string_view code, std::string_view data_json) override {
    // Time spent here is the log's, not the solve's (timing.solve_seconds).
    const auto start = std::chrono::steady_clock::now();
    if (comm_.rank() == 0) {
      dir_->append_log(log_line(level, source, message, code, data_json));
    } else if (level != "info") {
      Json data = Json::parse(data_json.empty() ? "{}" : data_json);
      data["rank"] = comm_.rank();
      dir_->append_log(log_line(level, source, message, code, data.dump()));
    }
    log_seconds_ += seconds_since(start);
  }

 private:
  // One log.jsonl line: wall-clock UTC time in milliseconds and the current
  // iteration (null before the first).
  [[nodiscard]] std::string log_line(std::string_view level, std::string_view source,
                                     std::string_view message, std::string_view code,
                                     std::string_view data_json) const {
    const auto now = std::chrono::system_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;
    char fraction[8];
    std::snprintf(fraction, sizeof fraction, ".%03dZ", static_cast<int>(ms));
    return Json{{"time", utc(now, "%Y-%m-%dT%H:%M:%S") + fraction},
                {"iteration", iteration_ > 0 ? Json(iteration_) : Json(nullptr)},
                {"level", level},
                {"source", source},
                {"message", message},
                {"code", code.empty() ? Json(nullptr) : Json(code)},
                {"data", Json::parse(data_json.empty() ? "{}" : data_json)}}
        .dump();
  }

  // Returns the final status; every value is a string literal.
  std::string_view iterate(std::int64_t& done) {
    const auto& in = p_.input;
    const bool steady = in.time.kind == "steady";
    const std::int64_t budget = steady ? in.time.max_iterations : in.time.steps;
    const auto equations = residual_names(in);
    // Steady convergence is measured against iteration 1 of the original run
    // (solver-input.md, `start`); a `continue` goes on counting and timing.
    std::vector<double> first;
    RoundOffExemption exempt(
        equations, continued() && in.start->woken ? *in.start->woken : std::vector<std::string>{});
    double time = 0.0, average_start = 0.0;
    std::int64_t from = 1;
    if (continued()) {
      const Start& s = *in.start;
      if (s.residual_reference) first = *s.residual_reference;
      if (s.time) time = *s.time;
      from = s.iterations + 1;
      if (!info_["averaging"].is_null()) average_start = info_["averaging"]["from_time"];
    }
    for (std::int64_t it = from; it <= budget; ++it) {
      // A launcher forwards a signal to every rank, not at the same instant,
      // and may abort the job on one: the stop file asks for the same stop
      // without signals. Rank 0 looks for it; the ranks stop together once
      // any has a reason to.
      std::int64_t stop = stop_requested;
      if (comm_.rank() == 0 && !stop_file_.empty()) {
        std::error_code ec;
        if (fs::exists(stop_file_, ec)) stop = 1;
      }
      if (comm_.max(stop) != 0) {
        stopped_by_signal_ = true;
        return "stopped";
      }
      iteration_ = it;
      // The solve: the physics' own work, without the history and log writes.
      const auto stepping = std::chrono::steady_clock::now();
      const double logged = log_seconds_;
      StepReport report = physics_->step(it);
      if (gpu_) (void)gpu_->sample();
      solve_seconds_ += seconds_since(stepping) - (log_seconds_ - logged);
      report.diverged_cell = first_diverged(report.diverged_cell);
      if (report.residuals.size() != equations.size()) {
        throw std::runtime_error("the physics reported " + std::to_string(report.residuals.size()) +
                                 " residuals for " + std::to_string(equations.size()) +
                                 " equations");
      }
      bool finite = !report.diverged_cell.has_value();
      Json residuals = Json::object();
      for (std::size_t e = 0; e < equations.size(); ++e) {
        finite = finite && std::isfinite(report.residuals[e]);
        residuals[equations[e]] =
            std::isfinite(report.residuals[e]) ? Json(report.residuals[e]) : Json(nullptr);
      }
      if (!finite) {
        diverge(it, report.diverged_cell);
        return "diverged";
      }
      // The step's time and averaging become run metadata only once its
      // history line is recorded, so that run.json never runs ahead of it.
      const double step_time = time + report.time_step;
      Json averaging = info_["averaging"];
      if (!steady && in.time.average_from && it > *in.time.average_from) {
        if (it == *in.time.average_from + 1) average_start = time;
        const auto averaging_start = std::chrono::steady_clock::now();
        const double logged_before = log_seconds_;
        physics_->accumulate_average(report.time_step);
        solve_seconds_ += seconds_since(averaging_start) - (log_seconds_ - logged_before);
        averaging = Json{{"from_step", *in.time.average_from + 1},
                         {"to_step", it},
                         {"from_time", average_start},
                         {"to_time", step_time}};
      }
      Json record{{"iteration", it},
                  {"time", steady ? Json(nullptr) : Json(step_time)},
                  {"residuals", residuals},
                  {"monitors", Json::object()}};
      for (std::size_t m = 0; m < in.monitors.size(); ++m) {
        const auto& monitor = in.monitors[m];
        if (it % monitor.every != 0) continue;
        record["monitors"][monitor.name] = monitor_record(m, "at iteration " + std::to_string(it));
      }
      if (comm_.rank() == 0) dir_->append_history(record.dump());
      // `iterations` is the last completed (recorded) iteration: a diverging
      // one is reported by its D diagnostic, not counted.
      done = it;
      time = step_time;
      info_["iterations"] = it;
      if (!steady) {
        info_["time"] = time;
        info_["averaging"] = averaging;
      }
      if (steady) {
        if (first.empty()) first = report.residuals;
        exempt.reference(first);
        // case-schema.md#time: equations at the round-off floor are exempt while they stay there
        // (runtime/round_off.hpp); one leaving it reports W0710, once.
        for (const std::size_t e : exempt.update(report.residuals)) {
          add_diagnostic(
              Diagnostic{.code = "W0710",
                         .message = "equation " + equations[e] + " left its floor at iteration " +
                                    std::to_string(it) + ": it now has to converge like the others",
                         .path = "",
                         .data_json = Json{{"equation", equations[e]},
                                           {"iteration", it},
                                           {"residual", report.residuals[e]},
                                           {"floor", exempt.floor()}}
                                          .dump()});
        }
        const double drop = std::pow(10.0, -in.time.residual_orders);
        bool converged = true;
        for (std::size_t e = 0; e < equations.size(); ++e) {
          const double r = report.residuals[e];
          converged = converged && (exempt.exempt(e) || r <= first[e] * drop ||
                                    (!floor_.empty() && r <= floor_[e]));
        }
        if (converged) return "converged";
      }
    }
    return "completed";
  }

  // The mesh index of the least diverged cell over all ranks, from the cell
  // of this rank's physics mesh (on a partition, a ghost names its owner).
  std::optional<index_t> first_diverged(std::optional<index_t> local) {
    constexpr std::int64_t none = std::numeric_limits<std::int64_t>::min();
    std::int64_t key = none;
    if (local) key = -1 - physics_mesh().cell_global_id()[at(*local)];
    key = comm_.max(key);
    if (key == none) return std::nullopt;
    return static_cast<index_t>(-1 - key);
  }

  void diverge(std::int64_t iteration, std::optional<index_t> cell) {
    nlohmann::json data{{"iteration", iteration}};
    std::string where;
    if (cell) {
      const std::string& zone = p_.mesh.cell_zones()[at(p_.mesh.cell_zone()[at(*cell)])].name;
      data["cell"] = *cell;
      data["zone"] = zone;
      where = " near cell " + std::to_string(*cell) + " (zone '" + zone + "')";
    }
    add_diagnostic(
        Diagnostic{.code = "D0107",
                   .message = "divergence at iteration " + std::to_string(iteration) + where,
                   .path = "",
                   .data_json = data.dump()});
  }

  // Collects the solution from the physics (PhysicsFailure on any error of
  // its members) and writes it; skips a diverged state that is not finite.
  void write_solution(bool diverged) {
    const auto& in = p_.input;
    io::Solution s;
    Json metadata = info_;
    for (const char* key :
         {"status", "finished", "iterations", "monitors", "timing", "diagnostics"})
      metadata.erase(key);
    // ASCII JSON, as readers decode CGNS descriptors.
    s.run_metadata = metadata.dump(-1, ' ', true);
    std::map<std::string, const BoundaryInput*> conditions;
    for (const auto& b : in.boundaries) conditions[b.zone] = &b;
    const auto offsets = p_.mesh.boundary_zone_offsets();
    for (const index_t z : io::meshinfo_zones(p_.mesh).boundary) {
      const auto it = conditions.find(p_.mesh.face_zones()[at(z)].name);
      s.boundary_bc_types.emplace_back(it == conditions.end() ? "" : bc_type(*it->second));
      if (it != conditions.end() && (it->second->kind == "wall" || it->second->kind == "slip")) {
        for (index_t f = offsets[at(z)]; f < offsets[at(z) + 1]; ++f) s.wall_faces.push_back(f);
      }
    }
    std::ranges::sort(s.wall_faces);
    // The physics names faces in its own mesh. On a partition, this rank's
    // wall faces are its owned boundary faces whose mesh face is a wall
    // face, ascending, and `slots` their positions in s.wall_faces.
    std::vector<index_t> physics_faces;
    std::vector<std::int64_t> slots;
    if (local_) {
      for (index_t f = local_->mesh.n_interior_faces(); f < local_->mesh.n_owned_faces(); ++f) {
        const auto global = static_cast<index_t>(local_->face_global_id[at(f)]);
        const auto it = std::ranges::lower_bound(s.wall_faces, global);
        if (it == s.wall_faces.end() || *it != global) continue;
        physics_faces.push_back(f);
        slots.push_back(it - s.wall_faces.begin());
      }
    } else {
      physics_faces = s.wall_faces;
      for (std::size_t k = 0; k < physics_faces.size(); ++k) slots.push_back(std::int64_t(k));
    }
    try {
      s.fields = physics_->fields(false);
      if (!info_["averaging"].is_null()) s.mean_fields = physics_->fields(true);
      s.restart_fields = physics_->restart_fields();
      if (!s.wall_faces.empty()) s.wall_fields = physics_->wall_fields(physics_faces);
    } catch (const std::exception& e) {
      throw PhysicsFailure(e.what());
    }
    std::vector<std::string> names;
    names.reserve(s.fields.size());
    for (const auto& f : s.fields) names.push_back(f.name);
    if (names != field_names(in))
      throw PhysicsFailure("the physics returned another set of fields");
    // Rank 0 gathers every field, cells by mesh index, wall faces by slot.
    const auto ids = owned_ids();
    const auto gather = [&](std::vector<io::Field>& fields, std::span<const std::int64_t> where,
                            std::int64_t n) {
      for (auto& f : fields) {
        if (f.values.size() != where.size()) {
          throw PhysicsFailure("the physics returned " + std::to_string(f.values.size()) +
                               " values of " + f.name + " for " + std::to_string(where.size()));
        }
        f.values = comm_.gather(where, f.values, n);
      }
    };
    // A failure before the gathers end leaves other ranks inside them: it
    // fails the run like a physics failure, which ends every rank.
    try {
      gather(s.fields, ids, p_.mesh.n_cells());
      gather(s.mean_fields, ids, p_.mesh.n_cells());
      gather(s.restart_fields, ids, p_.mesh.n_cells());
      gather(s.wall_fields, slots, static_cast<std::int64_t>(s.wall_faces.size()));
    } catch (const PhysicsFailure&) {
      throw;
    } catch (const std::exception& e) {
      throw PhysicsFailure(std::string("gathering the solution: ") + e.what());
    }
    if (comm_.rank() != 0) return;
    // A diverged state may hold non-finite values; results.md asks for its
    // solution only if one can be written, and readers require finite fields.
    if (diverged) {
      for (const auto& f : s.fields) {
        if (!std::ranges::all_of(f.values, [](double v) { return std::isfinite(v); })) return;
      }
    }
    io::write_solution(dir_->solution_staging(), p_.mesh, s);
    dir_->publish_solution();
  }

  // The forces of every monitor on the final state, whether or not its last
  // iteration was a multiple of `every` (results.md, run.json `monitors`).
  // The forces record of monitor m on the current state. A value that is not
  // finite fails the run (E0704) rather than reaching history.jsonl or
  // run.json as null (results.md, History).
  [[nodiscard]] Json monitor_record(std::size_t m, const std::string& when) const {
    const auto& in = p_.input;
    Json record = forces_record(physics_->loads(p_.monitor_zones[m], in.monitors[m].moment_center),
                                in.monitors[m], in.freestream);
    for (const auto& [key, value] : record.items()) {
      if (!std::isfinite(value.get<double>())) {
        throw std::runtime_error("monitor '" + in.monitors[m].name + "': " + key +
                                 " is not finite " + when);
      }
    }
    return record;
  }

  void record_final_monitors(std::int64_t done) {
    const auto& in = p_.input;
    Json monitors = Json::object();
    try {
      for (std::size_t m = 0; m < in.monitors.size(); ++m) {
        Json record = monitor_record(m, "on the final state");
        record["iteration"] = done;
        monitors[in.monitors[m].name] = std::move(record);
      }
    } catch (const std::exception& e) {
      throw PhysicsFailure(e.what());
    }
    info_["monitors"] = std::move(monitors);
  }

  static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  // results.md, run.json `timing`: the process's wall-clock seconds and those
  // of the physics' steps and averaging (the slowest rank's), for this run's
  // iterations; null without any. Collective.
  // The memory every rank used (#325): its host high-water mark and its
  // device memory beyond what was free at its start, at their peak and once
  // the physics was built, in bytes. Collective.
  void log_memory() {
    if (gpu_) (void)gpu_->sample();
    const auto n = at(comm_.size());
    const auto r = at(comm_.rank());
    std::vector<ExactSum> slots(4 * n);
    slots[4 * r].add(static_cast<double>(host_peak_bytes()));
    slots[4 * r + 1].add(static_cast<double>(gpu_ ? gpu_->peak() : 0));
    slots[4 * r + 2].add(static_cast<double>(host_after_setup_));
    slots[4 * r + 3].add(static_cast<double>(device_after_setup_));
    comm_.sum(slots);
    Json host = Json::array(), device = Json::array();
    std::int64_t host_setup = 0, device_setup = 0;
    for (std::size_t k = 0; k < n; ++k) {
      host.push_back(static_cast<std::int64_t>(slots[4 * k].value()));
      device.push_back(static_cast<std::int64_t>(slots[4 * k + 1].value()));
      host_setup = std::max(host_setup, static_cast<std::int64_t>(slots[4 * k + 2].value()));
      device_setup = std::max(device_setup, static_cast<std::int64_t>(slots[4 * k + 3].value()));
    }
    const auto most = [](const Json& values) {
      std::int64_t m = 0;
      for (const auto& v : values) m = std::max(m, v.get<std::int64_t>());
      return m;
    };
    note("info", "solver", "memory", "",
         Json{{"host_peak_bytes", most(host)},
              {"device_peak_bytes", most(device)},
              {"host_peak_bytes_by_rank", host},
              {"device_peak_bytes_by_rank", device},
              {"after_setup", {{"host", host_setup}, {"device", device_setup}}}}
             .dump());
  }

  Json timing(std::int64_t done) {
    const double solve = -comm_.min(-solve_seconds_);
    const std::int64_t iterations = done - (continued() ? p_.input.start->iterations : 0);
    if (iterations <= 0) return nullptr;
    return Json{
        {"wall_seconds", seconds_since(launched_)},
        {"solve_seconds", solve},
        {"iterations", iterations},
        {"ns_per_cell_iteration",
         solve * 1e9 / (static_cast<double>(p_.mesh.n_cells()) * static_cast<double>(iterations))}};
  }

  ExitStatus finish(std::string status, std::int64_t done) {
    info_["iterations"] = done;
    if (physics_ && status != "failed" && status != "diverged") {
      try {
        record_final_monitors(done);
      } catch (const PhysicsFailure& e) {
        // A throw on one rank leaves the others inside the collective sum of
        // loads(): abort them all, as for write_solution below.
        if (comm_.size() > 1) return fail(e.what(), done);
        record_failure(e.what());
        status = "failed";
      }
    }
    // results.md: the solution is required unless the run diverged (written
    // if possible) or failed (optional).
    if (physics_ && status != "failed") {
      try {
        write_solution(status == "diverged");
      } catch (const PhysicsFailure& e) {
        if (comm_.size() > 1) return fail(e.what(), done);
        record_failure(e.what());
        status = "failed";
      } catch (const std::exception& e) {
        if (status != "diverged") {
          record_failure(std::string("writing the solution: ") + e.what());
          status = "failed";
        }
      }
    }
    // Only rank 0 writes the solution: its status is the run's.
    status = comm_.broadcast(status);
    log_memory();
    // A failed run is not timed (results.md).
    info_["timing"] = status == "failed" ? Json(nullptr) : timing(done);
    if (status == "failed") info_["monitors"] = nullptr;
    info_["status"] = status;
    info_["finished"] = utc(std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%SZ");
    note(status == "failed" ? "error" : "info", "solver", "run finished: " + status, "",
         Json{{"status", status}, {"iterations", done}}.dump());
    if (comm_.rank() == 0) dir_->write_run_json(info_.dump(2));
    if (status == "failed") return ExitStatus::failed;
    return stopped_by_signal_ ? ExitStatus::signalled : ExitStatus::finished;
  }

  Prepared& p_;
  Comm& comm_;
  std::chrono::steady_clock::time_point launched_;  // run_case's entry
  double log_seconds_ = 0.0;                        // spent in write()
  fs::path stop_file_;                              // --stop-file; empty: none
  double solve_seconds_ = 0.0;                      // this rank's step() and accumulate_average()
  // Declared before the physics, so destroyed after it: its stream outlives the physics' use.
  std::optional<Gpu> gpu_;
  Geometry geometry_;
  CellGroups groups_;  // the physics' groups, over p_.mesh's cells
  std::optional<LocalPartition> local_;
  std::vector<std::int32_t> colours_;  // global colouring, the physics mesh's cells
  std::int32_t n_colours_ = 0;
  // The groups this rank owns, over the physics mesh's owned cells, and their colouring.
  std::vector<offset_t> group_offsets_;
  std::vector<index_t> group_cells_;
  std::vector<std::int32_t> group_colours_;
  std::int32_t n_group_colours_ = 0;
  std::unique_ptr<GhostExchange> ghosts_;
  std::optional<StartState> local_start_;                // the owned cells' share of p_.start
  std::optional<std::vector<io::Field>> local_initial_;  // and of p_.initial
  Json info_;
  std::unique_ptr<RunDirectory> dir_;
  std::unique_ptr<Physics> physics_;
  std::int64_t host_after_setup_ = 0;    // host_peak_bytes() once the physics exists
  std::int64_t device_after_setup_ = 0;  // the device's sample() then
  bool stopped_by_signal_ = false;
  std::int64_t iteration_ = 0;
  std::vector<double> floor_;  // fp32+acc64 convergence floor per equation; empty: none  // the
                               // iteration being computed, 0 before the first
};

}  // namespace

void PhysicsRegistry::add(PhysicsKey key, PhysicsFactory factory, PhysicsScope scope,
                          bool several_ranks, PartitionGroups groups) {
  entries_.emplace_back(std::move(key), Entry{std::move(factory), std::move(scope), several_ranks,
                                              std::move(groups)});
}

const PhysicsRegistry::Entry* PhysicsRegistry::find(const SolverInput& in) const {
  const auto matches = [](const std::string& pattern, const std::string& value) {
    return pattern == "*" || pattern == value;
  };
  for (const auto& [key, entry] : entries_) {
    if (matches(key.kind, in.physics_kind) && matches(key.turbulence, in.turbulence.model) &&
        matches(key.precision, in.precision) && matches(key.device, in.device_kind)) {
      return &entry;
    }
  }
  return nullptr;
}

ExitStatus run_case(const fs::path& input_file, const fs::path& runs,
                    const PhysicsRegistry& registry, const SolverBuild& build, std::FILE* out,
                    std::FILE* err, const fs::path& stop_file) {
  const auto launched = std::chrono::steady_clock::now();
  // Every rank reads and checks the same input, so all reach the same
  // verdict; rank 0 prints it.
  std::unique_ptr<Comm> comm;
  try {
    comm = make_comm();
  } catch (const Error& e) {
    std::fprintf(
        err, "%s\n",
        Diagnostic{
            .code = error_code_string(e.code()), .message = e.what(), .path = "", .data_json = "{}"}
            .json()
            .c_str());
    return ExitStatus::invalid_input;
  }
  // The ranks agree on whether the input can run before any collective
  // call; the first rank that cannot prints why: a refused input (exit 2)
  // or any other failure (E0704, exit 1).
  std::optional<Prepared> prepared;
  std::optional<Diagnostic> refusal;
  try {
    prepared.emplace(prepare(input_file, registry, comm->size()));
  } catch (const InputError& e) {
    refusal = e.diagnostic();
  } catch (const std::exception& e) {
    refusal = Diagnostic{.code = "E0704", .message = e.what(), .path = "", .data_json = "{}"};
  }
  // The key of the first refusing rank: -(2 rank + 1 if it failed).
  constexpr std::int64_t none = std::numeric_limits<std::int64_t>::min();
  const std::int64_t key =
      refusal ? -(2 * std::int64_t{comm->rank()} + (refusal->code == "E0704" ? 1 : 0)) : none;
  const std::int64_t first = comm->max(key);
  if (first != none) {
    if (first == key) std::fprintf(err, "%s\n", refusal->json().c_str());
    return -first % 2 == 1 ? ExitStatus::failed : ExitStatus::invalid_input;
  }
  stop_requested = 0;
  const SignalGuard signals;
  try {
    Run run(*prepared, build, *comm, launched);
    run.stop_on(stop_file);
    run.open(runs);
    if (comm->rank() == 0) {
      std::fprintf(out, "%s\n", run.path().string().c_str());
      std::fflush(out);
    }
    return run.execute();
  } catch (const std::exception& e) {
    // The run could not be set up (a coded error keeps its code, invalid
    // partition groups E0140), its directory created or its run.json written:
    // nothing records the failure but this diagnostic. On several ranks, the
    // others may wait in a collective call: every rank ends.
    // Every rank may fail here at once: each line goes out in one write, so
    // that the launcher's merged stderr keeps lines whole.
    const auto* coded = dynamic_cast<const Error*>(&e);
    const std::string line =
        Diagnostic{.code = coded != nullptr ? error_code_string(coded->code()) : "E0704",
                   .message = e.what(),
                   .path = "",
                   .data_json = Json{{"rank", comm->rank()}}.dump()}
            .json() +
        "\n";
    std::fwrite(line.data(), 1, line.size(), err);
    std::fflush(err);
    if (comm->size() > 1) comm->abort(static_cast<int>(ExitStatus::failed));
    return ExitStatus::failed;
  }
}

}  // namespace autan::runtime
