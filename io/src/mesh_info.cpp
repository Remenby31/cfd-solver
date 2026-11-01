#include "autan/io/mesh_info.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <picosha2.h>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/quality.hpp"
#include "cursor.hpp"

namespace autan::io {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

void hash_file(picosha2::hash256_one_by_one& hasher, const fs::path& path, bool framed) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error(ErrorCode::io_cannot_open, "cannot open '" + path.string() + "'");
  if (framed) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) throw Error(ErrorCode::io_cannot_open, "cannot stat '" + path.string() + "'");
    const std::string frame = path.filename().string() + "\n" + std::to_string(size) + "\n";
    hasher.process(frame.begin(), frame.end());
  }
  std::vector<char> buffer(std::size_t{1} << 20U);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hasher.process(buffer.begin(), buffer.begin() + in.gcount());
  }
  if (in.bad()) throw Error(ErrorCode::io_cannot_open, "cannot read '" + path.string() + "'");
}

const char* bc_type_name(BcKind kind) {
  switch (kind) {
    case BcKind::wall:
      return "wall";
    case BcKind::thermal_wall:
      return "thermal-wall";
    case BcKind::slip:
      return "slip";
    case BcKind::symmetry:
      return "symmetry";
    case BcKind::far_field:
      return "farfield";
    case BcKind::inlet:
      return "inlet";
    case BcKind::outlet:
      return "outlet";
    case BcKind::periodic:
      return "periodic";
    case BcKind::unspecified:
      return nullptr;
  }
  return nullptr;
}

Json name_or_null(const char* name) { return name == nullptr ? Json(nullptr) : Json(name); }
Json string_or_null(const std::string& s) { return s.empty() ? Json(nullptr) : Json(s); }

// A cell is one of the four standard shapes when its face sizes, vertex
// count and vertex degrees (faces per vertex) are that shape's: the only
// simple polyhedra with six quadrilaterals, or two triangles and three
// quadrilaterals, are the hexahedron and the triangular prism; a pyramid's
// apex is its one vertex of degree 4, off its quadrilateral.
std::size_t shape_index(const Mesh& mesh, index_t cell, std::vector<index_t>& nodes) {
  const auto faces = mesh.cell_faces(cell);
  int triangles = 0, quads = 0;
  nodes.clear();
  index_t quad_face = -1;
  for (const index_t f : faces) {
    const auto fn = mesh.face_nodes(f);
    if (fn.size() == 3) ++triangles;
    if (fn.size() == 4) {
      ++quads;
      quad_face = f;
    }
    nodes.insert(nodes.end(), fn.begin(), fn.end());
  }
  std::ranges::sort(nodes);
  std::vector<int> degrees;
  for (std::size_t i = 0; i < nodes.size();) {
    std::size_t j = i;
    while (j < nodes.size() && nodes[j] == nodes[i]) ++j;
    degrees.push_back(static_cast<int>(j - i));
    i = j;
  }
  const auto count = [&](int d) { return std::count(degrees.begin(), degrees.end(), d); };
  const std::size_t n = faces.size();
  const std::size_t v = degrees.size();
  if (n == 4 && triangles == 4 && v == 4) return 0;
  if (n == 5 && triangles == 4 && quads == 1 && v == 5 && count(3) == 4 && count(4) == 1) {
    for (const index_t p : mesh.face_nodes(quad_face)) {
      const auto range = std::ranges::equal_range(nodes, p);
      if (range.size() != 3) return 4;
    }
    return 1;
  }
  if (n == 5 && triangles == 2 && quads == 3 && v == 6 && count(3) == 6) return 2;
  if (n == 6 && quads == 6 && v == 8 && count(3) == 8) return 3;
  return 4;
}

}  // namespace

MeshInfoZones meshinfo_zones(const Mesh& mesh) {
  MeshInfoZones zones;
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    if (mesh.face_zones()[z].boundary) zones.boundary.push_back(static_cast<index_t>(z));
  }
  std::vector<char> used(mesh.cell_zones().size(), 0);
  for (const index_t z : mesh.cell_zone()) used[at(z)] = 1;
  for (std::size_t z = 0; z < used.size(); ++z) {
    if (used[z] != 0) zones.cell.push_back(static_cast<index_t>(z));
  }
  return zones;
}

std::string mesh_hash(const fs::path& path) {
  picosha2::hash256_one_by_one hasher;
  std::error_code ec;
  if (fs::is_directory(path, ec)) {
    const fs::path dir = polymesh_directory(path);
    for (const char* name : {"points", "faces", "owner", "neighbour", "boundary", "cellZones"}) {
      const fs::path file = dir / name;
      if (std::string_view(name) == "cellZones" && !fs::exists(file, ec)) continue;
      hash_file(hasher, file, true);
    }
  } else {
    hash_file(hasher, path, false);
  }
  hasher.finish();
  return picosha2::get_hash_hex_string(hasher);
}

std::optional<ZoneGroups> read_zone_groups(const fs::path& mesh_path, const Mesh& mesh,
                                           const std::string& sha256) {
  // Next to the mesh file itself, symlinks followed (case-schema.md), so the
  // SDK and the runtime read the same sidecar; for an OpenFOAM case
  // directory, next to its polyMesh directory.
  std::error_code ec;
  const fs::path given =
      fs::is_directory(mesh_path, ec) ? polymesh_directory(mesh_path) : mesh_path;
  const fs::path resolved = fs::canonical(given, ec);
  const fs::path path = (ec ? given.lexically_normal() : resolved).string() + ".zones.json";
  // Absent only when nothing is there: a link to nothing or an error is unreadable.
  const auto entry = fs::symlink_status(path, ec);
  if (ec && ec != std::errc::no_such_file_or_directory) {
    throw Error(ErrorCode::zone_groups_invalid, path.string() + ": " + ec.message());
  }
  if (!fs::exists(entry)) return std::nullopt;
  if (!fs::is_regular_file(path, ec)) {
    throw Error(ErrorCode::zone_groups_invalid, path.string() + ": not a readable file");
  }
  const auto invalid = [&](ErrorCode code, const std::string& why) -> Error {
    return {code, path.string() + ": " + why};
  };
  const std::ifstream in(path, std::ios::binary);
  std::ostringstream bytes;
  bytes << in.rdbuf();
  if (!in && !in.eof()) throw invalid(ErrorCode::zone_groups_invalid, "cannot be read");
  const std::string text = bytes.str();

  // Duplicate keys, which a JSON parser would merge silently: a group name
  // twice is E0139, any other key twice breaks the schema.
  std::vector<std::vector<std::string>> keys;
  std::string root_key;
  bool duplicate_group = false, duplicate_other = false;
  const auto watch = [&](int depth, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
    using E = nlohmann::json::parse_event_t;
    if (event == E::object_start) keys.emplace_back();
    if (event == E::object_end) keys.pop_back();
    if (event == E::key) {
      const std::string key = parsed.get<std::string>();
      if (depth == 1) root_key = key;
      auto& seen = keys.back();
      if (std::ranges::find(seen, key) != seen.end()) {
        (depth == 2 && root_key == "groups" ? duplicate_group : duplicate_other) = true;
      }
      seen.push_back(key);
    }
    return true;
  };
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(text, watch);
  } catch (const nlohmann::json::exception& e) {
    throw invalid(ErrorCode::zone_groups_invalid, std::string("not JSON: ") + e.what());
  }
  if (duplicate_other) throw invalid(ErrorCode::zone_groups_invalid, "a key is given twice");
  if (duplicate_group)
    throw invalid(ErrorCode::zone_groups_duplicate_name, "a group is given twice");
  const auto schema = [&](const std::string& why) {
    throw invalid(ErrorCode::zone_groups_invalid, why);
  };
  if (!doc.is_object()) schema("not an object");
  for (const auto& [key, value] : doc.items()) {
    if (key != "schema" && key != "mesh_sha256" && key != "provenance" && key != "groups")
      schema("unknown key '" + key + "'");
  }
  for (const char* key : {"schema", "mesh_sha256", "provenance", "groups"}) {
    if (!doc.contains(key)) schema(std::string("no '") + key + "'");
  }
  if (doc["schema"] != "autan.zone-groups/1") schema("schema is not autan.zone-groups/1");
  if (!doc["mesh_sha256"].is_string()) schema("mesh_sha256 is not a string");
  if (doc["mesh_sha256"].get<std::string>() != sha256) {
    throw invalid(
        ErrorCode::zone_groups_other_mesh,
        "mesh_sha256 " + doc["mesh_sha256"].get<std::string>() + " is not the mesh's " + sha256);
  }
  if (!doc["provenance"].is_object()) schema("provenance is not an object");
  if (!doc["groups"].is_object()) schema("groups is not an object");

  ZoneGroups out;
  out.path = path.string();
  out.sha256 = picosha2::hash256_hex_string(text);
  for (const auto& [key, value] : doc["provenance"].items()) {
    if (!value.is_string()) schema("provenance '" + key + "' is not a string");
    out.provenance.emplace_back(key, value.get<std::string>());
  }
  std::map<std::string, std::size_t> order;  // boundary zone name -> MeshInfo position
  const MeshInfoZones zones = meshinfo_zones(mesh);
  for (std::size_t k = 0; k < zones.boundary.size(); ++k)
    order[mesh.face_zones()[at(zones.boundary[k])].name] = k;
  const std::regex name("[A-Za-z0-9_.][A-Za-z0-9_.-]*");
  for (const auto& [group, members] : doc["groups"].items()) {
    const auto bad = [&](const std::string& why) {
      std::string message = "group '";
      message += group;
      message += "': ";
      message += why;
      throw invalid(ErrorCode::zone_groups_bad_group, message);
    };
    if (!std::regex_match(group, name)) bad("not a valid name");
    if (!members.is_array()) schema("group '" + group + "' is not a list");
    if (members.empty()) bad("empty");
    std::vector<std::pair<std::size_t, std::string>> found;
    for (const auto& zone : members) {
      if (!zone.is_string()) schema("group '" + group + "' lists a non-string");
      const auto it = order.find(zone.get<std::string>());
      if (it == order.end()) bad("no boundary zone '" + zone.get<std::string>() + "'");
      if (std::ranges::find(found, it->second, &std::pair<std::size_t, std::string>::first) !=
          found.end())
        bad("zone '" + it->first + "' listed twice");
      found.emplace_back(it->second, it->first);
    }
    std::ranges::sort(found);
    std::vector<std::string> names;
    names.reserve(found.size());
    for (const auto& [k, zone] : found) names.push_back(zone);
    out.groups.emplace_back(group, std::move(names));
  }
  std::ranges::sort(out.groups);
  return out;
}

std::string mesh_info_json(const Mesh& mesh, const std::string& sha256,
                           const std::optional<ZoneGroups>& groups) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw Error(ErrorCode::mesh_inconsistent_sizes, "MeshInfo describes an unpartitioned mesh");
  }
  const Geometry geometry = compute_geometry(mesh);
  const MeshQuality quality = compute_quality(mesh, geometry);

  std::array<std::int64_t, 5> by_shape{};
  std::vector<index_t> scratch;
  std::size_t max_faces = 0;
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    ++by_shape[shape_index(mesh, c, scratch)];
    max_faces = std::max(max_faces, mesh.cell_faces(c).size());
  }

  const MeshInfoZones zones = meshinfo_zones(mesh);
  Json boundary = Json::array();
  const auto offsets = mesh.boundary_zone_offsets();
  for (const index_t zi : zones.boundary) {
    const auto z = at(zi);
    const FaceZone& zone = mesh.face_zones()[z];
    boundary.push_back(Json{{"name", zone.name},
                            {"type", name_or_null(bc_type_name(zone.kind))},
                            {"native_type", string_or_null(zone.bc_type)},
                            {"faces", offsets[z + 1] - offsets[z]}});
  }

  std::vector<std::int64_t> cells_in_zone(mesh.cell_zones().size(), 0);
  for (const index_t z : mesh.cell_zone()) ++cells_in_zone[at(z)];
  Json cell_zones = Json::array();
  for (const index_t zi : zones.cell) {
    const auto z = at(zi);
    const CellZone& zone = mesh.cell_zones()[z];
    const char* type = zone.kind == CellKind::fluid   ? "fluid"
                       : zone.kind == CellKind::solid ? "solid"
                                                      : nullptr;
    cell_zones.push_back(Json{{"name", zone.name},
                              {"type", name_or_null(type)},
                              {"native_type", string_or_null(zone.type)},
                              {"cells", cells_in_zone[z]}});
  }

  std::map<std::pair<std::string, std::string>, std::int64_t> shared;
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    const auto& a = mesh.cell_zones()[at(mesh.cell_zone()[at(mesh.owner()[at(f)])])].name;
    const auto& b = mesh.cell_zones()[at(mesh.cell_zone()[at(mesh.neighbour()[at(f)])])].name;
    if (a != b) ++shared[std::minmax(a, b)];
  }
  Json interfaces = Json::array();
  for (const auto& [pair, faces] : shared) {
    interfaces.push_back(Json{{"zones", {pair.first, pair.second}}, {"faces", faces}});
  }

  Json periodic = Json::array();
  for (const auto& pair : mesh.periodic_pairs()) {
    periodic.push_back(
        Json{{"zones",
              {mesh.face_zones()[at(pair.zone_a)].name, mesh.face_zones()[at(pair.zone_b)].name}},
             {"transform",
              {{"kind", "translation"},
               {"vector", {pair.translation[0], pair.translation[1], pair.translation[2]}}}},
             {"faces", offsets[at(pair.zone_a) + 1] - offsets[at(pair.zone_a)]}});
  }

  Json group_list = Json::array();
  Json source = nullptr;
  if (groups) {
    for (const auto& [name, zones_of] : groups->groups)
      group_list.push_back(Json{{"name", name}, {"zones", zones_of}});
    Json provenance = Json::object();
    for (const auto& [key, value] : groups->provenance) provenance[key] = value;
    source = Json{{"path", groups->path}, {"sha256", groups->sha256}, {"provenance", provenance}};
  }
  const double max_angle = *std::ranges::max_element(quality.non_orthogonality);
  const auto [x_low, x_high] = std::ranges::minmax(mesh.x());
  const auto [y_low, y_high] = std::ranges::minmax(mesh.y());
  const auto [z_low, z_high] = std::ranges::minmax(mesh.z());
  const Json doc{
      {"schema", "autan.mesh-info/1"},
      {"source",
       {{"path", mesh.source().path}, {"format", mesh.source().format}, {"sha256", sha256}}},
      {"counts",
       {{"cells", mesh.n_cells()},
        {"faces", mesh.n_faces()},
        {"nodes", mesh.n_points()},
        {"cells_by_type",
         {{"tetra", by_shape[0]},
          {"pyramid", by_shape[1]},
          {"prism", by_shape[2]},
          {"hexa", by_shape[3]},
          {"polyhedron", by_shape[4]}}}}},
      {"bounds", {{"min", {x_low, y_low, z_low}}, {"max", {x_high, y_high, z_high}}}},
      {"boundary_zones", boundary},
      {"cell_zones", cell_zones},
      {"interfaces", interfaces},
      {"periodic", periodic},
      {"groups", group_list},
      {"groups_source", source},
      {"quality", {{"max_non_orthogonality_deg", max_angle}, {"max_faces_per_cell", max_faces}}},
  };
  return doc.dump(2);
}

}  // namespace autan::io
