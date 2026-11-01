#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "fixtures.hpp"

namespace {

namespace fs = std::filesystem;
const fs::path data_dir{AUTAN_TEST_MESH_DIR};

fs::path fresh_dir(const std::string& name) {
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_mesh_info-" + std::to_string(random())) / name;
  fs::create_directories(dir);
  return dir;
}

void write(const fs::path& path, const std::string& content) {
  std::ofstream(path, std::ios::binary) << content;
}

}  // namespace

TEST_CASE("The mesh hash follows the MeshInfo contract", "[mesh-info]") {
  const fs::path dir = fresh_dir("hash");
  SECTION("file: SHA-256 of its bytes (FIPS 180-2 test vector)") {
    write(dir / "abc.msh", "abc");
    CHECK(autan::io::mesh_hash(dir / "abc.msh") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  }
  // Expected digests computed with Python's hashlib over the framed
  // concatenation "<name>\n<length>\n<bytes>" in contract order.
  SECTION("polyMesh directory without cellZones") {
    write(dir / "points", "p");
    write(dir / "faces", "ff");
    write(dir / "owner", "");
    write(dir / "neighbour", "n");
    write(dir / "boundary", "b");
    CHECK(autan::io::mesh_hash(dir) ==
          "a60f55a0d1f0eedb2434572069b71f36320c3d949b334dc1eadbd136651e1ea8");
    write(dir / "cellZones", "cz");
    CHECK(autan::io::mesh_hash(dir) ==
          "3d0fa5c0ad6c894d6f887b1a3ad8ba0418a4b06e5208125c29f0ae91073ee9df");
  }
}

TEST_CASE("MeshInfo of the hybrid test mesh has its constructed counts", "[mesh-info]") {
  // tests/data/mesh/generate/generate.py, hybrid with n = 4: 8 hexahedra and
  // 8 cubes of 6 pyramids (lower corner block), 2 prisms per cube in the 16
  // cubes of each of two blocks, 6 tetrahedra per cube in the last 16. Zones
  // lower (k < 2) and upper share the plane z = 1/2: 8 quadrilaterals and 16
  // triangles. Side x = 0: 8 quadrilaterals and 16 triangles.
  const std::string source = GENERATE("hybrid/hybrid.ascii.msh", "hybrid/polyMesh");
  CAPTURE(source);
  const fs::path path = data_dir / source;
  const auto doc = nlohmann::json::parse(
      autan::io::mesh_info_json(autan::io::read_mesh(path), autan::io::mesh_hash(path)));

  CHECK(doc["schema"] == "autan.mesh-info/1");
  CHECK(doc["source"]["sha256"] == autan::io::mesh_hash(path));
  CHECK(doc["counts"]["cells"] == 216);
  const auto& types = doc["counts"]["cells_by_type"];
  CHECK(types["tetra"] == 96);
  CHECK(types["pyramid"] == 48);
  CHECK(types["prism"] == 64);
  CHECK(types["hexa"] == 8);
  CHECK(types["polyhedron"] == 0);
  // The unit cube of generate.py, in the file's units.
  CHECK(doc["bounds"]["min"] == nlohmann::json::array({0.0, 0.0, 0.0}));
  CHECK(doc["bounds"]["max"] == nlohmann::json::array({1.0, 1.0, 1.0}));
  REQUIRE(doc["boundary_zones"].size() == 6);
  CHECK(doc["boundary_zones"][0]["name"] == "xmin");
  CHECK(doc["boundary_zones"][0]["faces"] == 24);
  REQUIRE(doc["cell_zones"].size() == 2);
  REQUIRE(doc["interfaces"].size() == 1);
  CHECK(doc["interfaces"][0]["zones"] == nlohmann::json::array({"lower", "upper"}));
  CHECK(doc["interfaces"][0]["faces"] == 24);
  CHECK(doc["periodic"] == nlohmann::json::array());
  CHECK(doc["quality"]["max_faces_per_cell"] == 6);
  int cells = 0;
  for (const auto& zone : doc["cell_zones"]) cells += zone["cells"].get<int>();
  CHECK(cells == 216);
}

TEST_CASE("MeshInfo types follow the normalisation table", "[mesh-info]") {
  const fs::path path = data_dir / "hanging" / "hanging.fluent.msh";
  const auto doc = nlohmann::json::parse(
      autan::io::mesh_info_json(autan::io::read_mesh(path), autan::io::mesh_hash(path)));
  CHECK(doc["source"]["format"] == "fluent");
  CHECK(doc["counts"]["cells_by_type"]["polyhedron"] == 1);
  CHECK(doc["counts"]["cells_by_type"]["hexa"] == 100);
  CHECK(doc["quality"]["max_faces_per_cell"] == 105);
  for (const auto& zone : doc["boundary_zones"]) {
    CAPTURE(zone.dump());
    if (zone["name"] == "walls") {
      CHECK(zone["type"] == "wall");
    } else {
      CHECK(zone["type"] == "outlet");  // foamMeshToFluent writes patches as pressure-outlet
      CHECK(zone["native_type"] == "pressure-outlet");
    }
  }
  CHECK(doc["cell_zones"][0]["type"] == "fluid");
}

TEST_CASE("Only cells with the topology of a standard shape count as that shape", "[mesh-info]") {
  // A triangular prism with one rectangular side split by a diagonal and the
  // two cap edges it ends on split at their midpoints: six quadrilaterals and
  // eight vertices, but a vertex of degree 4, so not a hexahedron.
  autan::CellwiseMesh cw;
  const std::array<std::array<double, 3>, 8> points{
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {0.5, 0, 0}, {0.5, 0, 1}}};
  for (const auto& p : points) {
    cw.x.push_back(p[0]);
    cw.y.push_back(p[1]);
    cw.z.push_back(p[2]);
  }
  const std::vector<autan::offset_t> offsets{0, 4, 8, 12, 16, 20, 24};
  const std::vector<autan::index_t> faces{0, 2, 1, 6, 3, 7, 4, 5, 0, 6, 1, 4,
                                          0, 4, 7, 3, 1, 2, 5, 4, 2, 0, 3, 5};
  cw.add_polyhedron(offsets, faces, 0);
  cw.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cw));
  const auto doc = nlohmann::json::parse(autan::io::mesh_info_json(mesh, std::string(64, '0')));
  CHECK(doc["counts"]["cells_by_type"]["hexa"] == 0);
  CHECK(doc["counts"]["cells_by_type"]["polyhedron"] == 1);
}

TEST_CASE("MeshInfo omits cell zones without cells", "[mesh-info]") {
  const fs::path dir = fresh_dir("polyMesh");
  for (const auto& entry : fs::directory_iterator(data_dir / "hex" / "polyMesh")) {
    fs::copy(entry.path(), dir / entry.path().filename());
  }
  std::string labels;
  for (int c = 0; c < 27; ++c) labels += std::to_string(c) + " ";
  write(dir / "cellZones",
        "FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }\n"
        "2 ( fluid { type cellZone; cellLabels List<label> 27(" +
            labels + "); } unused { type cellZone; cellLabels List<label> 0(); } )\n");
  const auto doc = nlohmann::json::parse(
      autan::io::mesh_info_json(autan::io::read_mesh(dir), autan::io::mesh_hash(dir)));
  REQUIRE(doc["cell_zones"].size() == 1);
  CHECK(doc["cell_zones"][0]["name"] == "fluid");
}

TEST_CASE("MeshInfo lists periodic pairs and types their zones periodic", "[mesh-info]") {
  // The unit box cut in 3^3 hexahedra, periodic along x: 9 faces a side.
  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::hexa);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const auto doc = nlohmann::json::parse(autan::io::mesh_info_json(
      autan::assemble(autan::to_face_based(std::move(cells))), std::string(64, '0')));
  REQUIRE(doc["periodic"].size() == 1);
  CHECK(doc["periodic"][0] ==
        nlohmann::json{{"zones", {"xmin", "xmax"}},
                       {"transform", {{"kind", "translation"}, {"vector", {1.0, 0.0, 0.0}}}},
                       {"faces", 9}});
  for (const auto& zone : doc["boundary_zones"]) {
    CAPTURE(zone.dump());
    const bool periodic = zone["name"] == "xmin" || zone["name"] == "xmax";
    CHECK((zone["type"] == "periodic") == periodic);
  }
}

TEST_CASE("A zone-groups sidecar gives MeshInfo its groups, checked", "[mesh-info][groups]") {
  // The hybrid mesh copied with a sidecar next to it (mesh-info.md, Zone
  // groups): a Gmsh file's is <file>.zones.json, a polyMesh directory's
  // <directory>.zones.json.
  const bool directory = GENERATE(false, true);
  CAPTURE(directory);
  const fs::path dir = fresh_dir("groups");
  const fs::path mesh_path = directory ? dir / "polyMesh" : dir / "hybrid.ascii.msh";
  fs::copy(directory ? data_dir / "hybrid" / "polyMesh" : data_dir / "hybrid" / "hybrid.ascii.msh",
           mesh_path, fs::copy_options::recursive);
  const fs::path sidecar = mesh_path.string() + ".zones.json";
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const std::string hash = autan::io::mesh_hash(mesh_path);
  const auto info = [&] {
    return nlohmann::json::parse(
        autan::io::mesh_info_json(mesh, hash, autan::io::read_zone_groups(mesh_path, mesh, hash)));
  };

  // No sidecar: no group.
  CHECK(info()["groups"] == nlohmann::json::array());
  CHECK(info()["groups_source"].is_null());

  // Groups sorted by name, overlapping, zones in MeshInfo order whatever the
  // file's; the source is the sidecar's path and the hash of its bytes.
  nlohmann::json doc{
      {"schema", "autan.zone-groups/1"},
      {"mesh_sha256", hash},
      {"provenance", {{"url", "https://example.org/grid"}, {"note", "test"}}},
      {"groups",
       {{"walls", {"zmin", "ymin"}}, {"x.sides", {"xmax", "xmin"}}, {"all-x_1", {"xmin"}}}}};
  write(sidecar, doc.dump());
  const auto with = info();
  CHECK(with["groups"] == nlohmann::json::parse(R"([
    {"name": "all-x_1", "zones": ["xmin"]},
    {"name": "walls", "zones": ["ymin", "zmin"]},
    {"name": "x.sides", "zones": ["xmin", "xmax"]}])"));
  CHECK(with["groups_source"]["path"] == fs::canonical(mesh_path).string() + ".zones.json");
  // Given as its OpenFOAM case directory, the polyMesh directory's sidecar.
  if (directory) {
    const fs::path case_dir = dir / "case";
    fs::create_directories(case_dir / "constant");
    fs::rename(mesh_path, case_dir / "constant" / "polyMesh");
    fs::rename(sidecar, case_dir / "constant" / "polyMesh.zones.json");
    CHECK(nlohmann::json::parse(autan::io::mesh_info_json(
              mesh, hash, autan::io::read_zone_groups(case_dir, mesh, hash)))["groups"] ==
          with["groups"]);
    fs::rename(case_dir / "constant" / "polyMesh", mesh_path);
    fs::rename(case_dir / "constant" / "polyMesh.zones.json", sidecar);
  }
  // Through a symlink to the mesh, the sidecar next to the mesh file itself.
  const fs::path link = dir / "link";
  fs::create_symlink(mesh_path, link);
  CHECK(nlohmann::json::parse(autan::io::mesh_info_json(
            mesh, hash, autan::io::read_zone_groups(link, mesh, hash)))["groups"] ==
        with["groups"]);
  CHECK(with["groups_source"]["sha256"] == autan::io::mesh_hash(sidecar));
  CHECK(with["groups_source"]["provenance"] == doc["provenance"]);

  // Each fault, with its code.
  const auto refused = [&](const std::string& text, autan::ErrorCode code) {
    write(sidecar, text);
    try {
      (void)autan::io::read_zone_groups(mesh_path, mesh, hash);
      FAIL("expected an error for " << text);
    } catch (const autan::Error& e) {
      CHECK(e.code() == code);
    }
  };
  const auto edited = [&](const std::function<void(nlohmann::json&)>& edit) {
    nlohmann::json d = doc;
    edit(d);
    return d.dump();
  };
  using autan::ErrorCode;
  refused("{not json", ErrorCode::zone_groups_invalid);
  refused(edited([](auto& d) { d["extra"] = 1; }), ErrorCode::zone_groups_invalid);
  refused(edited([](auto& d) { d.erase("provenance"); }), ErrorCode::zone_groups_invalid);
  refused(edited([](auto& d) { d["schema"] = "autan.zone-groups/2"; }),
          ErrorCode::zone_groups_invalid);
  refused(edited([](auto& d) { d["groups"]["walls"] = "ymin"; }), ErrorCode::zone_groups_invalid);
  refused(edited([](auto& d) { d["mesh_sha256"] = std::string(64, '0'); }),
          ErrorCode::zone_groups_other_mesh);
  refused(edited([](auto& d) { d["groups"]["walls"] = {"ymin", "nowhere"}; }),
          ErrorCode::zone_groups_bad_group);
  refused(edited([](auto& d) { d["groups"]["walls"] = nlohmann::json::array(); }),
          ErrorCode::zone_groups_bad_group);
  refused(edited([](auto& d) { d["groups"]["walls"] = {"ymin", "ymin"}; }),
          ErrorCode::zone_groups_bad_group);
  for (const char* name : {"-walls", "wing*", "a b", ""}) {
    CAPTURE(name);
    refused(edited([&](auto& d) { d["groups"][name] = {"ymin"}; }),
            ErrorCode::zone_groups_bad_group);
  }
  // A sidecar that is there but cannot be read: a link to itself.
  fs::remove(sidecar);
  fs::create_symlink(sidecar.filename(), sidecar);
  try {
    (void)autan::io::read_zone_groups(mesh_path, mesh, hash);
    FAIL("expected E0136 for a looping link");
  } catch (const autan::Error& e) {
    CHECK(e.code() == ErrorCode::zone_groups_invalid);
  }
  fs::remove(sidecar);
  // A name given twice, which a JSON parser would merge.
  refused(R"({"schema": "autan.zone-groups/1", "mesh_sha256": ")" + hash +
              R"(", "provenance": {}, "groups": {"walls": ["ymin"], "walls": ["zmin"]}})",
          ErrorCode::zone_groups_duplicate_name);
}
