// The Euler physics through autan-run's public entry point, run_case, on tests/data/mesh/hybrid:
// solver inputs written by `autan prepare` (tests/data/runtime), edited per test.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cgnslib.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#ifdef AUTAN_MPIEXEC
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#include "autan/core/geometry.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/physics/euler_physics.hpp"
#include "autan/physics/manufactured.hpp"
#include "autan/runtime/run.hpp"

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using autan::runtime::ExitStatus;

const fs::path data_dir{AUTAN_TEST_DATA_DIR};
const fs::path mesh_path = data_dir / "mesh" / "hybrid" / "polyMesh";
const fs::path mms_path = data_dir / ".." / ".." / "py" / "tests" / "fixtures" / "euler.mms.json";
const autan::runtime::SolverBuild build{.version = "0.0.0", .commit = std::string(40, 'a')};

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

Json input(const char* file, const std::function<void(Json&)>& edit) {
  Json doc = Json::parse(read(data_dir / "runtime" / file));
  doc["mesh"]["path"] = fs::absolute(mesh_path).string();
  if (doc["verification"].is_object())
    doc["verification"]["path"] = fs::absolute(mms_path).string();
  edit(doc);
  return doc;
}

struct Outcome {
  ExitStatus status;
  fs::path run;
  fs::path runs;  // the runs directory passed to run_case
  std::string err;
};

Outcome run(const Json& doc) {
  std::random_device random;
  const fs::path dir = fs::temp_directory_path() / ("autan_euler-" + std::to_string(random()));
  fs::create_directories(dir);
  std::ofstream(dir / "solver-input.json") << doc.dump();
  autan::runtime::PhysicsRegistry registry;
  autan::physics::register_euler(registry);
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  const ExitStatus status =
      autan::runtime::run_case(dir / "solver-input.json", dir / "runs", registry, build, out, err);
  auto text = [](std::FILE* f) {
    std::rewind(f);
    std::string s;
    for (int c = std::fgetc(f); c != EOF; c = std::fgetc(f)) s.push_back(static_cast<char>(c));
    std::fclose(f);
    return s;
  };
  std::string printed = text(out);
  if (!printed.empty() && printed.back() == '\n') printed.pop_back();
  return Outcome{.status = status, .run = printed, .runs = dir / "runs", .err = text(err)};
}

std::vector<Json> history(const fs::path& run) {
  std::vector<Json> lines;
  std::ifstream in(run / "history.jsonl");
  for (std::string line; std::getline(in, line);) lines.push_back(Json::parse(line));
  return lines;
}

// The `start` the SDK writes to continue `previous` (solver-input.md).
Json continue_from(const fs::path& previous, const std::string& mode = "continue") {
  const Json info = Json::parse(read(previous / "run.json"));
  std::string hash;
  picosha2::hash256_hex_string(read(previous / "solution.cgns"), hash);
  Json reference = nullptr;
  Json woken = nullptr;
  if (info["time"].is_null() && mode == "continue") {
    // case-schema.md#time: the equations below 1e-12 of the largest at
    // iteration 1 that went above that floor.
    const auto lines = history(previous);
    reference = lines[0]["residuals"];
    double largest = 0.0;
    for (const auto& [name, f] : reference.items()) largest = std::max(largest, f.get<double>());
    std::vector<std::string> names;
    for (const auto& [name, f] : reference.items()) {
      const bool left = std::ranges::any_of(lines, [&](const Json& line) {
        return line["residuals"][name].get<double>() > 1e-12 * largest;
      });
      if (f.get<double>() < 1e-12 * largest && left) names.push_back(name);
    }
    std::ranges::sort(names);
    woken = Json(names);
  }
  return Json{{"run", previous.string()},
              {"run_id", info["run_id"]},
              {"identity", info["identity"]},
              {"mode", mode},
              {"iterations", info["iterations"]},
              {"time", info["time"]},
              {"solution_sha256", hash},
              {"residual_reference", reference},
              {"woken", woken}};
}

// Field `name` of the cell-centred FlowSolution_t `solution` of a run's solution.cgns.
std::vector<double> field(const fs::path& run, const char* name,
                          const char* solution = "FlowSolution") {
  int fn = 0;
  REQUIRE(cg_open((run / "solution.cgns").string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  char zone_name[33];
  cgsize_t size[3];
  REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
  int count = 0;
  REQUIRE(cg_nsols(fn, 1, 1, &count) == CG_OK);
  std::vector<double> values;
  for (int s = 1; s <= count; ++s) {
    char sol_name[33];
    CGNS_ENUMT(GridLocation_t) location{};
    REQUIRE(cg_sol_info(fn, 1, 1, s, sol_name, &location) == CG_OK);
    if (std::string(sol_name) != solution) continue;
    values.resize(static_cast<std::size_t>(size[1]));
    const cgsize_t first = 1;
    REQUIRE(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &size[1],
                          values.data()) == CG_OK);
  }
  cg_close(fn);
  REQUIRE(!values.empty());
  return values;
}

// The case of hybrid.solver-input.json made inviscid, with the freestream along x so that it is
// tangent to the slip wall ymin and the symmetry plane zmax; farfield elsewhere.
void inviscid_along_x(Json& doc, const char* precision) {
  doc["precision"] = precision;
  auto& t = doc["physics"]["turbulence"];
  t["model"] = "inviscid";
  t["nu_tilde_ratio"] = nullptr;
  auto& f = doc["freestream"];
  f["velocity"] = {f["speed"], 0.0, 0.0};
  f["aoa"] = 0.0;
  f["nu_tilde"] = nullptr;
  for (auto& z : doc["cell_zones"]) {
    z["initial"]["velocity"] = f["velocity"];
    z["initial"]["nu_tilde"] = nullptr;
  }
  for (auto& b : doc["boundaries"]) {
    if (b["kind"] == "wall") b["kind"] = "slip";
  }
}

}  // namespace

TEST_CASE("a uniform flow along slip walls and farfield faces stays uniform", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    // fp32+acc64 starts at its convergence floor (round-off only) and stops at iteration 1.
    d["monitors"][0]["every"] = 1;
    if (steady) {
      d["time"]["max_iterations"] = 20;
    } else {
      d["time"] = {
          {"kind", "unsteady"}, {"steps", 20}, {"dt", nullptr}, {"cfl", 0.8}, {"average_from", 10}};
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["diagnostics"].empty());
  const auto& f = doc["freestream"];
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double speed = f["speed"].get<double>();
  const double p = f["pressure"].get<double>();
  const double rho = f["density"].get<double>();
  // Each step changes the state by round-off of the face fluxes: 20 steps stay within 64 eps.
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) CHECK(std::abs(v - speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityZ")) CHECK(std::abs(v) <= 64 * eps * speed);
  // The pressure on the slip wall is the freestream's: no force.
  const auto lines = history(out.run);
  const auto& forces = lines.back()["monitors"]["forces"];
  for (const char* k : {"fx", "fy", "fz"}) CHECK(std::abs(forces[k].get<double>()) <= 64 * eps * p);
  if (!steady) CHECK(info["time"].get<double>() > 0.0);
  // case-schema.md#time: a steady fp32+acc64 run converges at 0.5 * 2^-23 * Phi_k at the
  // latest, Phi_k = f_k sqrt(sum_i A_i^2 / N) from the freestream flux f_k and the cell surfaces.
  if (!steady || std::string(precision) == "fp64") {
    CHECK(info["convergence"].is_null());
    return;
  }
  CHECK(info["status"] == "converged");
  const auto mesh = autan::io::read_mesh(mesh_path);
  const auto g = autan::compute_geometry(mesh);
  std::vector<double> surface(static_cast<std::size_t>(mesh.n_cells()), 0.0);
  for (std::size_t face = 0; face < static_cast<std::size_t>(mesh.n_faces()); ++face) {
    surface[static_cast<std::size_t>(mesh.owner()[face])] += g.face_area[face];
    if (face < static_cast<std::size_t>(mesh.n_interior_faces()))
      surface[static_cast<std::size_t>(mesh.neighbour()[face])] += g.face_area[face];
  }
  double sum = 0.0;
  for (const double a : surface) sum += a * a;
  const double area = std::sqrt(sum / static_cast<double>(surface.size()));
  const double gamma = doc["cell_zones"][0]["medium"]["gamma"].get<double>();
  const double c = std::sqrt(gamma * p / rho);
  const double q = speed + c;
  const double h = c * c / (gamma - 1.0) + 0.5 * speed * speed;
  const double unit = 0.5 * 0x1p-23 * area;
  const Json expected = {{"density", unit * rho * q},
                         {"momentum-x", unit * rho * q * q},
                         {"momentum-y", unit * rho * q * q},
                         {"momentum-z", unit * rho * q * q},
                         {"energy", unit * rho * h * q}};
  const auto& floor = info["convergence"]["floor"];
  REQUIRE(floor.size() == expected.size());
  for (const auto& [name, value] : expected.items()) {
    INFO(name);
    CHECK(std::abs(floor[name].get<double>() - value.get<double>()) <= 1e-12 * value.get<double>());
  }
}

TEST_CASE("a verification case converges to its discrete solution", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const char* linear = GENERATE("multigrid", "gauss-seidel");
  INFO(precision << " " << linear);
  const auto doc = input("hybrid-verification.solver-input.json", [&](Json& d) {
    d["precision"] = precision;
    d["time"]["linear_solver"] = linear;
    d["time"]["max_iterations"] = 20000;
    d["time"]["residual_orders"] = std::string(precision) == "fp64" ? 10.0 : 5.0;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(fs::exists(out.run / "manufactured.json"));
}

TEST_CASE("a continued run reproduces the uninterrupted one bit for bit", "[euler]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  auto doc = [&](int budget) {
    return input("hybrid-verification.solver-input.json", [&](Json& d) {
      d["precision"] = precision;
      if (steady) {
        d["time"]["max_iterations"] = budget;
        d["time"]["residual_orders"] = 30.0;
      } else {
        d["verification"] = nullptr;
        for (auto& b : d["boundaries"]) b["kind"] = "farfield";
        // Two cell zones at different states: a nonuniform, evolving flow.
        auto& init = d["cell_zones"].back()["initial"];
        init["density"] = 1.2 * init["density"].get<double>();
        init["pressure"] = 1.3 * init["pressure"].get<double>();
        d["time"] = {{"kind", "unsteady"},
                     {"steps", budget},
                     {"dt", nullptr},
                     {"cfl", 0.8},
                     {"average_from", 2}};
      }
    });
  };
  const auto whole = run(doc(12));
  REQUIRE(whole.status == ExitStatus::finished);
  const auto first = run(doc(6));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(12);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  auto lines = [](const fs::path& f) {
    std::vector<std::string> out;
    std::ifstream in(f);
    for (std::string l; std::getline(in, l);) out.push_back(l);
    return out;
  };
  auto chain = lines(first.run / "history.jsonl");
  const auto more = lines(second.run / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    CHECK(field(second.run, name) == field(whole.run, name));
    if (!steady) {
      CHECK(field(second.run, name, "FlowSolutionMean") ==
            field(whole.run, name, "FlowSolutionMean"));
    }
  }
  for (const char* name : {"Density", "MomentumX", "EnergyStagnationDensity"}) {
    CHECK(field(second.run, name, "RestartState") == field(whole.run, name, "RestartState"));
  }
  if (!steady) {
    CHECK(field(second.run, "AverageWeight", "RestartState") ==
          field(whole.run, "AverageWeight", "RestartState"));
  }
}

TEST_CASE("a run continued right after a shortened step reproduces the uninterrupted one",
          "[euler]") {
  // A shortened step caps the next CFL number; the cap is solver state and must cross a restart.
  // Two zones at very different states make the first steps shorten.
  auto doc = [&](int budget) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 3.0 * init["density"].get<double>();
      init["pressure"] = 5.0 * init["pressure"].get<double>();
      d["time"]["max_iterations"] = budget;
      d["time"]["residual_orders"] = 30.0;
    });
  };
  auto lines = [](const fs::path& f) {
    std::vector<std::string> out;
    std::ifstream in(f);
    for (std::string l; std::getline(in, l);) out.push_back(l);
    return out;
  };
  constexpr int budget = 40;
  const auto whole = run(doc(budget));
  REQUIRE(whole.status == ExitStatus::finished);
  int shortened = 0;
  for (const auto& l : lines(whole.run / "log.jsonl")) {
    const auto e = Json::parse(l);
    if (e["level"] == "warning" && e["source"] == "solver" && e["iteration"].is_number()) {
      shortened = e["iteration"].get<int>();
      break;
    }
  }
  INFO("first shortened step " << shortened);
  REQUIRE(shortened > 0);
  REQUIRE(shortened < budget);
  const auto first = run(doc(shortened));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(budget);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  REQUIRE(second.status == ExitStatus::finished);
  auto chain = lines(first.run / "history.jsonl");
  const auto more = lines(second.run / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "Pressure"}) {
    CHECK(field(second.run, name) == field(whole.run, name));
  }
}

TEST_CASE("a steady run reports the residual of its second-order scheme from the start",
          "[euler]") {
  // The steady solver starts at first order, but convergence is judged on the history, which
  // must hold the second-order residual: at iteration 1 it equals the unsteady run's, which is
  // second order from its first step. Two zones at different states make the flow nonuniform.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](bool steady) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, precision);
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      if (steady) {
        d["time"]["max_iterations"] = 1;
      } else {
        d["time"] = {{"kind", "unsteady"},
                     {"steps", 1},
                     {"dt", nullptr},
                     {"cfl", 0.8},
                     {"average_from", nullptr}};
      }
    });
  };
  const auto steady = run(doc(true));
  const auto unsteady = run(doc(false));
  REQUIRE(steady.status == ExitStatus::finished);
  REQUIRE(unsteady.status == ExitStatus::finished);
  CHECK(history(steady.run).front()["residuals"] == history(unsteady.run).front()["residuals"]);
}

namespace {

// tests/data/mesh/periodic-box, periodic in x and y: zmin a slip wall, zmax a symmetry plane,
// one cell zone; the case of hybrid.solver-input.json otherwise, inviscid.
void periodic_box(Json& d, const char* precision) {
  const fs::path box = data_dir / "mesh" / "periodic-box" / "periodic-box.ascii.msh";
  inviscid_along_x(d, precision);
  d["mesh"]["path"] = fs::absolute(box).string();
  d["mesh"]["sha256"] = autan::io::mesh_hash(box);
  d["periodic"] =
      Json::array({Json{{"zones", {"xmin", "xmax"}},
                        {"transform", {{"kind", "translation"}, {"vector", {1.0, 0.0, 0.0}}}}},
                   Json{{"zones", {"ymin", "ymax"}},
                        {"transform", {{"kind", "translation"}, {"vector", {0.0, 1.0, 0.0}}}}}});
  // Periodic zones take no condition.
  Json kept = Json::array();
  for (const auto& b : d["boundaries"]) {
    if (b["zone"] == "ymin") {
      Json wall = b;
      wall["zone"] = "zmin";
      kept.push_back(wall);
    }
    if (b["zone"] == "zmax") kept.push_back(b);
  }
  d["boundaries"] = kept;
  d["monitors"][0]["zones"] = {"zmin"};
  Json zone = d["cell_zones"][0];
  zone["zone"] = "fluid";
  d["cell_zones"] = Json::array({zone});
}

// A flow along x and y through both periodic pairs, `steps` SSP-RK3 steps.
void unsteady_through_pairs(Json& d, int steps) {
  auto& f = d["freestream"];
  const double speed = f["speed"].get<double>();
  f["velocity"] = {0.6 * speed, 0.8 * speed, 0.0};
  for (auto& z : d["cell_zones"]) z["initial"]["velocity"] = f["velocity"];
  d["time"] = {{"kind", "unsteady"},
               {"steps", steps},
               {"dt", nullptr},
               {"cfl", 0.8},
               {"average_from", nullptr}};
}

}  // namespace

TEST_CASE("a steady multigrid case with periodic pairs is refused before any directory exists",
          "[euler]") {
  // The multigrid agglomerates one mesh's cells and takes no image ghosts: E0706 at
  // `time.linear_solver`.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) { periodic_box(d, "fp64"); });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["location"]["path"] == "time.linear_solver");
}

TEST_CASE("a case with an initial-state file starts from the file's states", "[euler]") {
  // initial-state.md: the file gives every cell the freestream while both cell zones start at
  // twice its density. Started from the file, the flow is uniform with farfield all around and
  // stays so; started from the zones, it would not.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  Json doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    for (auto& z : d["cell_zones"]) {
      z["initial"]["density"] = 2.0 * d["freestream"]["density"].get<double>();
    }
    d["time"] = {{"kind", "unsteady"},
                 {"steps", 3},
                 {"dt", nullptr},
                 {"cfl", 0.8},
                 {"average_from", nullptr}};
  });
  const auto& f = doc["freestream"];
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  Json centroids = Json::array(), values = Json::array();
  for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_cells()); ++c) {
    centroids.push_back({g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]});
    values.push_back({rho, speed, 0.0, 0.0, p});
  }
  const Json state{{"schema", "autan.initial-state/1"},
                   {"oracle", "freestream"},
                   {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
                   {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
                   {"cells", mesh.n_cells()},
                   {"centroids", centroids},
                   {"values", values}};
  std::random_device random;
  const fs::path file = fs::temp_directory_path() / ("autan_initial-" + std::to_string(random()));
  std::ofstream(file) << state.dump();
  std::string sha;
  picosha2::hash256_hex_string(read(file), sha);
  doc["initial_state"] = {{"path", file.string()}, {"sha256", sha}};
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) {
    CHECK(std::abs(v - speed) <= 64 * eps * speed);
  }
}

TEST_CASE("a uniform flow through periodic pairs stays uniform", "[euler]") {
  // Image ghosts hold their owners' values: a uniform flow crossing both pairs is steady.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    periodic_box(d, precision);
    unsteady_through_pairs(d, 20);
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const auto& f = doc["freestream"];
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX"))
    CHECK(std::abs(v - 0.6 * speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityY"))
    CHECK(std::abs(v - 0.8 * speed) <= 64 * eps * speed);
}

TEST_CASE("cases outside the Euler physics are refused before any directory exists", "[euler]") {
  // E0706 on stderr, exit 2, no run directory; data names the zone and what it holds.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const std::string refused =
      GENERATE("solid", "second gas", "transport", "wall", "sideslip", "sonic", "unsteady vortex");
  INFO(precision << " " << refused);
  std::string path = "cell_zones[1].medium";
  std::string field = "medium";
  Json value = "helium";
  std::size_t boundary = 2;
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    auto& z = d["cell_zones"][1];
    if (refused == "sideslip" || refused == "sonic" || refused == "unsteady vortex") {
      // A point vortex on the farfield xmin, from the lift of the slip wall ymin.
      boundary = 0;
      d["boundaries"][0]["point_vortex"] = {{"zones", {"ymin"}}, {"center", {0.5, 0.0, -1.0}}};
    }
    if (refused == "solid") {
      z["medium"] = {{"kind", "solid"},
                     {"name", "steel"},
                     {"density", 7800.0},
                     {"specific_heat", 500.0},
                     {"conductivity", 40.0}};
      z["initial"] = {{"kind", "solid"}, {"temperature", 300.0}};
      value = "solid";
    } else if (refused == "second gas") {
      z["medium"]["name"] = "helium";
      z["medium"]["gas_constant"] = 2077.1;
      z["medium"]["gamma"] = 5.0 / 3.0;
    } else if (refused == "transport") {
      z["medium"]["name"] = "helium";
      z["medium"]["prandtl"] = 0.67;
    } else if (refused == "wall") {
      d["boundaries"][2]["kind"] = "wall";
      path = "boundaries[2].kind";
      field = "condition";
      value = "wall";
    } else if (refused == "sideslip") {
      d["freestream"]["sideslip"] = 0.1;
      path = "freestream.sideslip";
      field = "sideslip";
      value = 0.1;
    } else if (refused == "sonic") {
      d["freestream"]["mach"] = 1.0;
      path = "freestream.mach";
      field = "mach";
      value = 1.0;
    } else {
      d["time"] = {{"kind", "unsteady"},
                   {"steps", 2},
                   {"dt", nullptr},
                   {"cfl", 0.8},
                   {"average_from", nullptr}};
      path = "time.kind";
      field = "time";
      value = "unsteady";
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(out.run.empty());
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["data"]["path"] == path);
  CHECK(diagnostic["data"][field] == value);
  CHECK(diagnostic["data"]["zone"] ==
        (field == "medium" ? doc["cell_zones"][1]["zone"] : doc["boundaries"][boundary]["zone"]));
}

#ifdef AUTAN_MPIEXEC
namespace {

// autan-run under the launcher on `ranks` ranks: its run directory, none for an invalid input;
// its standard error in `err` when given.
fs::path run_on_ranks(const Json& doc, int ranks, int expected_status = 0,
                      std::string* err = nullptr) {
  std::random_device random;
  const fs::path dir = fs::temp_directory_path() / ("autan_euler_mpi-" + std::to_string(random()));
  fs::create_directories(dir);
  const std::string err_file = (dir / "stderr").string();
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, 2, err_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                                   0644);
  // solver-input.md, `devices`: one rank per device, so the input counts the ranks.
  Json with_count = doc;
  if (with_count["devices"]["count"] == 1) with_count["devices"]["count"] = ranks;
  std::ofstream(dir / "solver-input.json") << with_count.dump();
  std::vector<std::string> args{AUTAN_MPIEXEC,
                                AUTAN_MPIEXEC_NUMPROC_FLAG,
                                std::to_string(ranks),
                                AUTAN_RUN,
                                "--input",
                                (dir / "solver-input.json").string(),
                                "--runs",
                                (dir / "runs").string()};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = 0;
  REQUIRE(posix_spawn(&pid, AUTAN_MPIEXEC, &actions, nullptr, argv.data(), environ) == 0);
  posix_spawn_file_actions_destroy(&actions);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  if (err != nullptr) *err = read(err_file);
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == expected_status);
  fs::path run;
  if (fs::exists(dir / "runs")) {
    for (const auto& entry : fs::directory_iterator(dir / "runs")) run = entry.path();
  }
  return run;
}

}  // namespace

TEST_CASE("several ranks give the one-rank solution to round-off", "[euler][mpi]") {
  // runtime.md, Ranks: Euler on N ranks equals one rank to round-off (faces summed in local
  // order, issue #199); at a fixed rank count, runs repeat bit for bit. Two zones at different
  // states on the hybrid mesh (farfield around), or a flow through the periodic box's pairs.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool periodic = GENERATE(false, true);
  const int ranks = GENERATE(2, 4);
  CAPTURE(precision, periodic, ranks);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    if (periodic) {
      // Through both pairs and into the slip wall zmin, so that the flow evolves.
      periodic_box(d, precision);
      unsteady_through_pairs(d, 12);
      const double speed = d["freestream"]["speed"].get<double>();
      d["cell_zones"][0]["initial"]["velocity"] = {0.6 * speed, 0.7 * speed, -0.3 * speed};
    } else {
      inviscid_along_x(d, precision);
      for (auto& b : d["boundaries"]) b["kind"] = "farfield";
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["time"] = {{"kind", "unsteady"},
                   {"steps", 12},
                   {"dt", nullptr},
                   {"cfl", 0.8},
                   {"average_from", nullptr}};
    }
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, ranks);
  REQUIRE(!many.empty());
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  // History: every residual norm.
  const auto h1 = history(one.run), hn = history(many);
  REQUIRE(h1.size() == hn.size());
  double worst_norm = 0.0;
  for (std::size_t i = 0; i < h1.size(); ++i) {
    for (const auto& [name, v] : h1[i]["residuals"].items()) {
      const double a = v.get<double>(), b = hn[i]["residuals"][name].get<double>();
      worst_norm = std::max(worst_norm, std::abs(a - b) / (std::max(std::abs(a), 1e-300) * eps));
    }
  }
  // Fields: the largest difference over the field's largest magnitude (the largest speed for
  // the velocity components).
  const auto magnitude = [](const std::vector<double>& v) {
    double m = 0.0;
    for (const double x : v) m = std::max(m, std::abs(x));
    return m;
  };
  double speed = 0.0;
  {
    const auto vx = field(one.run, "VelocityX"), vy = field(one.run, "VelocityY"),
               vz = field(one.run, "VelocityZ");
    for (std::size_t c = 0; c < vx.size(); ++c) {
      speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
    }
  }
  double worst_field = 0.0;
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(one.run, name), b = field(many, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) diff = std::max(diff, std::abs(a[c] - b[c]));
    const double scale = std::string(name).starts_with("Velocity") ? speed : magnitude(a);
    worst_field = std::max(worst_field, diff / (scale * eps));
  }
  // Measured at most 7.5 eps on norms and 11.3 eps on fields over these 12 steps (fp64, 4 ranks).
  INFO("norms " << worst_norm << " eps, fields " << worst_field << " eps");
  CHECK(worst_norm <= 256.0);
  CHECK(worst_field <= 256.0);
  // A fixed rank count repeats bit for bit.
  const fs::path again = run_on_ranks(doc, ranks);
  CHECK(read(again / "history.jsonl") == read(many / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "Pressure"}) {
    CHECK(field(again, name) == field(many, name));
  }
}

TEST_CASE("a steady multigrid case on several ranks is refused before any directory exists",
          "[euler][mpi]") {
  // The multigrid agglomerates one mesh's cells: E0706 at `time.linear_solver`, exit 2.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) { inviscid_along_x(d, "fp64"); });
  std::string err;
  CHECK(run_on_ranks(doc, 2, 2, &err).empty());
  INFO(err);
  // The launcher may write its own lines; the diagnostic is the first JSON object.
  Json diagnostic;
  std::istringstream lines(err);
  for (std::string line; std::getline(lines, line);) {
    if (line.starts_with('{')) {
      diagnostic = Json::parse(line);
      break;
    }
  }
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["location"]["path"] == "time.linear_solver");
}

TEST_CASE("a steady run whose state is not physical on some ranks stops every rank",
          "[euler][mpi]") {
  // Zone `lower` at 1e-9 times the freestream density rounds to zero density in fp32+acc64. On 4
  // ranks some parts hold none of it: they must take the divergence branch with the others, or the
  // ranks enter different collectives and the run hangs instead of reporting D0107.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp32+acc64");
    for (auto& z : d["cell_zones"]) {
      if (z["zone"] == "lower")
        z["initial"]["density"] = 1e-9 * d["freestream"]["density"].get<double>();
    }
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 5;
  });
  const fs::path many = run_on_ranks(doc, 4);
  REQUIRE(!many.empty());
  const Json info = Json::parse(read(many / "run.json"));
  INFO(info["diagnostics"].dump());
  CHECK(info["status"] == "diverged");
  const auto& diagnostics = info["diagnostics"];
  const auto d107 = std::find_if(diagnostics.begin(), diagnostics.end(),
                                 [](const Json& g) { return g["code"] == "D0107"; });
  REQUIRE(d107 != diagnostics.end());
  CHECK((*d107)["data"]["zone"] == "lower");
}

TEST_CASE("a steady Gauss-Seidel case on several ranks converges to the one-rank solution",
          "[euler][mpi]") {
  // Gauss-Seidel sweeps the runtime's colours by global id, and the step length, the worst cell
  // and the norms reduce over the ranks, so N ranks run the one-rank algorithm. The face sums
  // still follow local order (#199), so the bits differ: converged 10 orders, the fields agree
  // within 1e-9 of their magnitude (the largest speed for velocity components), and a fixed rank
  // count repeats bit for bit. Two zones at different states on the hybrid mesh, farfield around.
  const int ranks = GENERATE(2, 4);
  CAPTURE(ranks);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    auto& init = d["cell_zones"].back()["initial"];
    init["density"] = 1.2 * init["density"].get<double>();
    init["pressure"] = 1.3 * init["pressure"].get<double>();
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 2000;
    d["time"]["residual_orders"] = 10.0;
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  CHECK(Json::parse(read(one.run / "run.json"))["status"] == "converged");
  const fs::path many = run_on_ranks(doc, ranks);
  REQUIRE(!many.empty());
  CHECK(Json::parse(read(many / "run.json"))["status"] == "converged");
  const auto vx = field(one.run, "VelocityX"), vy = field(one.run, "VelocityY"),
             vz = field(one.run, "VelocityZ");
  double speed = 0.0;
  for (std::size_t c = 0; c < vx.size(); ++c) {
    speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
  }
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(one.run, name), b = field(many, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0, scale = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) {
      diff = std::max(diff, std::abs(a[c] - b[c]));
      scale = std::max(scale, std::abs(a[c]));
    }
    if (std::string(name).starts_with("Velocity")) scale = speed;
    INFO(name << " differs by " << diff / scale << " of its magnitude");
    CHECK(diff <= 1e-9 * scale);
  }
  const fs::path again = run_on_ranks(doc, ranks);
  CHECK(read(again / "history.jsonl") == read(many / "history.jsonl"));
}

TEST_CASE("steady Gauss-Seidel steps through periodic pairs follow the one-rank steps",
          "[euler][mpi]") {
  // The periodic box: image ghosts take du after each colour. Not converged: 40 steps, the
  // fields agree within 1e-9 of their magnitude.
  const int ranks = GENERATE(2, 4);
  CAPTURE(ranks);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    periodic_box(d, "fp64");
    const double speed = d["freestream"]["speed"].get<double>();
    d["cell_zones"][0]["initial"]["velocity"] = {0.6 * speed, 0.7 * speed, -0.3 * speed};
    d["time"]["linear_solver"] = "gauss-seidel";
    d["time"]["max_iterations"] = 40;
    d["time"]["residual_orders"] = 30.0;
  });
  const auto one = run(doc);
  INFO(one.err);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, ranks);
  REQUIRE(!many.empty());
  const auto vx = field(one.run, "VelocityX"), vy = field(one.run, "VelocityY"),
             vz = field(one.run, "VelocityZ");
  double speed = 0.0;
  for (std::size_t c = 0; c < vx.size(); ++c) {
    speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
  }
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(one.run, name), b = field(many, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0, scale = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) {
      diff = std::max(diff, std::abs(a[c] - b[c]));
      scale = std::max(scale, std::abs(a[c]));
    }
    if (std::string(name).starts_with("Velocity")) scale = speed;
    INFO(name << " differs by " << diff / scale << " of its magnitude");
    CHECK(diff <= 1e-9 * scale);
  }
}
#endif

TEST_CASE("an equation at the round-off floor does not set the steady CFL number", "[euler]") {
  // tests/data/mesh/ringleb is one cell thick in z: the z-momentum is identically zero and its
  // residual round-off. Two runs differ only by a uniform initial w of 1e-20 times the speed,
  // which changes that round-off and the other equations by an ulp at most; the CFL law must
  // ignore it (case-schema.md#time, the runtime's exemption), so over 40 steps the other
  // equations' histories agree within 1e-6 (measured 8.5e-9; driven by the round-off, they
  // differ by 3.4).
  const fs::path mesh = data_dir / "mesh" / "ringleb" / "ringleb-n004.msh";
  auto doc = [&](double w) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      d["mesh"]["path"] = fs::absolute(mesh).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
      const double speed = d["freestream"]["speed"].get<double>();
      Json zone = d["cell_zones"][0];
      zone["zone"] = "fluid";
      zone["initial"]["pressure"] = 1.3 * zone["initial"]["pressure"].get<double>();
      zone["initial"]["velocity"] = {speed, 0.0, w * speed};
      d["cell_zones"] = Json::array({zone});
      Json boundaries = Json::array();
      for (const char* name : {"k-min", "k-max", "inflow", "outflow", "back", "front"}) {
        Json b = d["boundaries"][0];
        for (auto& [key, value] : b.items()) {
          if (key != "zone" && key != "kind") value = nullptr;
        }
        b["zone"] = name;
        const bool side = std::string(name) != "back" && std::string(name) != "front";
        b["kind"] = side ? "farfield" : "symmetry";
        boundaries.push_back(b);
      }
      d["boundaries"] = boundaries;
      d["monitors"] = Json::array();
      d["time"]["max_iterations"] = 40;
      d["time"]["residual_orders"] = 30.0;
    });
  };
  const auto a = run(doc(0.0));
  INFO(a.err);
  REQUIRE(a.status == ExitStatus::finished);
  const auto b = run(doc(1e-20));
  REQUIRE(b.status == ExitStatus::finished);
  const auto ha = history(a.run), hb = history(b.run);
  REQUIRE(ha.size() == hb.size());
  bool noisy = false;
  double worst = 0.0;
  for (std::size_t i = 0; i < ha.size(); ++i) {
    for (const char* k : {"density", "momentum-x", "momentum-y", "energy"}) {
      const double x = ha[i]["residuals"][k].get<double>();
      const double y = hb[i]["residuals"][k].get<double>();
      worst = std::max(worst, std::abs(x - y) / std::abs(x));
    }
    noisy = noisy || hb[i]["residuals"]["momentum-z"] != ha[i]["residuals"]["momentum-z"];
  }
  INFO("histories differ by " << worst);
  CHECK(worst <= 1e-6);
  // A continuation carries the exemption: from 20 steps, without `start.woken` it repeats the
  // uninterrupted run; with the z-momentum woken, that equation's round-off sets the CFL again.
  auto half = doc(1e-20);
  half["time"]["max_iterations"] = 20;
  const auto first = run(half);
  REQUIRE(first.status == ExitStatus::finished);
  auto rest = doc(1e-20);
  rest["start"] = continue_from(first.run);
  REQUIRE(rest["start"]["woken"] == Json::array());
  const auto plain = run(rest);
  INFO(plain.err);
  REQUIRE(plain.status == ExitStatus::finished);
  CHECK(read(first.run / "history.jsonl") + read(plain.run / "history.jsonl") ==
        read(b.run / "history.jsonl"));
  rest["start"]["woken"] = {"momentum-z"};
  const auto woken = run(rest);
  REQUIRE(woken.status == ExitStatus::finished);
  CHECK(read(woken.run / "history.jsonl") != read(plain.run / "history.jsonl"));
  CHECK(noisy);  // the injected noise did reach the z-momentum residual
}

TEST_CASE("a diverged run names the cell whose stored state is not physical", "[euler]") {
  // Zone `lower` starts at 1e-9 times the freestream density: FP32 stores it as a difference from
  // the freestream that rounds to zero density, so the residual is not finite and the run
  // diverges; D0107 must name a cell of that zone. In FP64 the state is representable; whatever
  // happens, a divergence names its cell.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    for (auto& z : d["cell_zones"]) {
      if (z["zone"] == "lower")
        z["initial"]["density"] = 1e-9 * d["freestream"]["density"].get<double>();
    }
    d["time"]["max_iterations"] = 5;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(!out.run.empty());  // a run directory, whatever the exit status
  const Json info = Json::parse(read(out.run / "run.json"));
  INFO(info["diagnostics"].dump());
  if (std::string(precision) == "fp32+acc64") {
    CHECK(info["status"] == "diverged");
  } else {
    CHECK(info["status"] == "completed");  // FP64 represents the state and runs on
  }
  if (info["status"] == "diverged") {
    const auto& diagnostics = info["diagnostics"];
    const auto d107 = std::find_if(diagnostics.begin(), diagnostics.end(),
                                   [](const Json& g) { return g["code"] == "D0107"; });
    REQUIRE(d107 != diagnostics.end());
    CHECK((*d107)["data"]["zone"] == "lower");
  }
}

TEST_CASE("a fast flow at low pressure that the kernels resolve is not taken for divergence",
          "[euler]") {
  // rho = 1, |v| = 1e4, p = 0.7: the absolute energy rounds the pressure away in FP32, but the
  // stored differences and the gauge pressure keep it; the check must judge the state as the
  // kernels read it.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    auto& f = d["freestream"];
    const double r = d["cell_zones"][0]["medium"]["gas_constant"].get<double>();
    f["density"] = 1.0;
    f["pressure"] = 0.7;
    f["temperature"] = 0.7 / r;
    f["speed"] = 1e4;
    f["velocity"] = {1e4, 0.0, 0.0};
    f["mach"] = 1e4 / std::sqrt(1.4 * 0.7);
    for (auto& z : d["cell_zones"]) {
      z["initial"]["density"] = 1.0;
      z["initial"]["pressure"] = 0.7;
      z["initial"]["temperature"] = 0.7 / r;
      z["initial"]["velocity"] = f["velocity"];
    }
    for (auto& b : d["boundaries"]) b["kind"] = "farfield";
    d["time"] = {{"kind", "unsteady"},
                 {"steps", 3},
                 {"dt", nullptr},
                 {"cfl", 0.5},
                 {"average_from", nullptr}};
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  CHECK(Json::parse(read(out.run / "run.json"))["status"] == "completed");
}

TEST_CASE("oracle-state faces take their paired states, whatever the row order", "[euler]") {
  // tests/data/mesh/ringleb with the exact Ringleb states (autan-vv) on its four in-plane sides,
  // in the oracle's units (R = 1/gamma); the same rows in reverse order within each zone give
  // the same solution bit for bit, because faces pair with rows by centroid (face-states.md).
  const fs::path dir = data_dir / "mesh" / "ringleb";
  auto doc_for = [&](bool reversed) {
    Json states = Json::parse(read(dir / "ringleb-n004.states.json"));
    if (reversed) {
      for (auto& [name, z] : states["zones"].items()) {
        std::reverse(z["centroids"].begin(), z["centroids"].end());
        std::reverse(z["values"].begin(), z["values"].end());
      }
    }
    const std::string bytes = states.dump();
    std::random_device random;
    const fs::path file =
        fs::temp_directory_path() / ("autan_ringleb-" + std::to_string(random()) + ".json");
    std::ofstream(file, std::ios::binary) << bytes;
    std::string sha;
    picosha2::hash256_hex_string(bytes, sha);
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, "fp64");
      const fs::path mesh = dir / "ringleb-n004.msh";
      d["mesh"]["path"] = fs::absolute(mesh).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
      const double gamma = 1.4, rho = 0.8, p = 0.6, r = 1.0 / gamma;
      auto& f = d["freestream"];
      f["density"] = rho;
      f["pressure"] = p;
      f["temperature"] = p / (rho * r);
      f["speed"] = 0.5;
      f["velocity"] = {0.0, -0.5, 0.0};
      f["mach"] = 0.5 / std::sqrt(gamma * p / rho);
      Json zone = d["cell_zones"][0];
      zone["zone"] = "fluid";
      zone["medium"]["gas_constant"] = r;
      zone["initial"]["density"] = rho;
      zone["initial"]["pressure"] = p;
      zone["initial"]["temperature"] = p / (rho * r);
      zone["initial"]["velocity"] = f["velocity"];
      d["cell_zones"] = Json::array({zone});
      Json boundaries = Json::array();
      for (const char* name : {"k-min", "k-max", "inflow", "outflow", "back", "front"}) {
        Json b = d["boundaries"][0];
        for (auto& [key, value] : b.items()) {
          if (key != "zone" && key != "kind") value = nullptr;
        }
        b["zone"] = name;
        const bool side = std::string(name) != "back" && std::string(name) != "front";
        b["kind"] = side ? "oracle-state" : "symmetry";
        if (side) b["states"] = Json{{"path", file.string()}, {"sha256", sha}};
        boundaries.push_back(b);
      }
      d["boundaries"] = boundaries;
      d["monitors"][0]["zones"] = {"k-min"};
      d["time"]["max_iterations"] = 200;
    });
  };
  const auto a = run(doc_for(false));
  INFO(a.err);
  REQUIRE(a.status == ExitStatus::finished);
  const auto b = run(doc_for(true));
  REQUIRE(b.status == ExitStatus::finished);
  for (const char* name : {"Density", "VelocityX", "VelocityY", "Pressure"}) {
    CHECK(field(a.run, name) == field(b.run, name));
  }
  // The states reach the cells: the flow is no longer the uniform initial state.
  const auto rho = field(a.run, "Density");
  CHECK(*std::ranges::max_element(rho) - *std::ranges::min_element(rho) > 0.1);
}

TEST_CASE("oracle-state faces impose their states component by component", "[euler]") {
  // Every side of the hybrid mesh takes one uniform state with all five components distinct and
  // non-zero; started there, the flow stays there in both precisions.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  const std::array<double, 5> w{1.2, 80.0, -30.0, 20.0, 1.1e5};
  Json zones = Json::object();
  const auto off = mesh.boundary_zone_offsets();
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    Json centroids = Json::array(), values = Json::array();
    for (auto f = off[z]; f < off[z + 1]; ++f) {
      const auto i = static_cast<std::size_t>(f);
      centroids.push_back({g.face_centroid_x[i], g.face_centroid_y[i], g.face_centroid_z[i]});
      values.push_back(w);
    }
    if (!centroids.empty()) {
      zones[mesh.face_zones()[z].name] = {
          {"faces", centroids.size()}, {"centroids", centroids}, {"values", values}};
    }
  }
  const Json states = {
      {"schema", "autan.face-states/1"},
      {"oracle", "uniform"},
      {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
      {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
      {"zones", zones}};
  const std::string bytes = states.dump();
  std::random_device random;
  const fs::path file =
      fs::temp_directory_path() / ("autan_uniform-" + std::to_string(random()) + ".json");
  std::ofstream(file, std::ios::binary) << bytes;
  std::string sha;
  picosha2::hash256_hex_string(bytes, sha);
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    for (auto& z : d["cell_zones"]) {
      z["initial"]["density"] = w[0];
      z["initial"]["velocity"] = {w[1], w[2], w[3]};
      z["initial"]["pressure"] = w[4];
    }
    for (auto& b : d["boundaries"]) {
      b["kind"] = "oracle-state";
      b["states"] = Json{{"path", file.string()}, {"sha256", sha}};
    }
    d["time"]["max_iterations"] = 20;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double speed = std::hypot(w[1], w[2], w[3]);
  const std::array<std::pair<const char*, double>, 5> expected{{{"Density", w[0]},
                                                                {"VelocityX", w[1]},
                                                                {"VelocityY", w[2]},
                                                                {"VelocityZ", w[3]},
                                                                {"Pressure", w[4]}}};
  for (const auto& [name, value] : expected) {
    INFO(name);
    const double scale =
        std::string(name) == "Density" || std::string(name) == "Pressure" ? std::abs(value) : speed;
    for (const double v : field(out.run, name)) CHECK(std::abs(v - value) <= 64 * eps * scale);
  }
}

TEST_CASE("a point vortex takes the Kutta-Joukowski circulation of its surfaces' lift", "[euler]") {
  // tests/data/mesh/hybrid with y halved (span 0.5); the slip walls xmin and zmin carry the
  // vortex of every other side. Uniform at p1 = 1.3 p_inf, the walls carry (p1 - p_inf) A on
  // their outward normals -x and -z (A = 0.5 each), so with e_L = (-sin a, 0, cos a) the lift per
  // unit span is L' = (p1 - p_inf)(sin a - cos a), and the first step's residual gives
  // Gamma = L' / (rho_inf V_inf).
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  std::random_device random;
  const fs::path mesh = fs::temp_directory_path() / ("autan_halfspan-" + std::to_string(random()));
  fs::create_directories(mesh);
  for (const auto& entry : fs::directory_iterator(mesh_path)) {
    fs::copy_file(entry.path(), mesh / entry.path().filename());
  }
  {
    std::istringstream in(read(mesh_path / "points"));
    std::ofstream out(mesh / "points", std::ios::trunc);
    for (std::string line; std::getline(in, line);) {
      double x = 0, y = 0, z = 0;
      if (std::sscanf(line.c_str(), "(%lf %lf %lf)", &x, &y, &z) == 3) {
        char text[128];
        std::snprintf(text, sizeof text, "(%.17g %.17g %.17g)", x, 0.5 * y, z);
        line = text;
      }
      out << line << '\n';
    }
  }
  const double a = 0.2;
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    d["mesh"]["path"] = mesh.string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
    auto& f = d["freestream"];
    const double speed = f["speed"].get<double>();
    f["aoa"] = a;
    f["velocity"] = {speed * std::cos(a), 0.0, speed * std::sin(a)};
    const double r = d["cell_zones"][0]["medium"]["gas_constant"].get<double>();
    const double p1 = 1.3 * f["pressure"].get<double>();
    for (auto& z : d["cell_zones"]) {
      z["initial"]["velocity"] = f["velocity"];
      z["initial"]["pressure"] = p1;
      z["initial"]["temperature"] = p1 / (z["initial"]["density"].get<double>() * r);
    }
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "xmin" || b["zone"] == "zmin") {
        b["kind"] = "slip";
      } else {
        b["kind"] = "farfield";
        b["point_vortex"] = {{"zones", {"xmin", "zmin"}}, {"center", {0.5, 0.0, -0.5}}};
      }
    }
    d["monitors"] = Json::array();
    d["time"]["max_iterations"] = 1;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const auto& f = doc["freestream"];
  const double p = f["pressure"].get<double>(), rho = f["density"].get<double>();
  const double speed = f["speed"].get<double>();
  const double expected = 0.3 * p * (std::sin(a) - std::cos(a)) / (rho * speed);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double circulation = field(out.run, "VortexCirculation0", "RestartState").front();
  CHECK(std::abs(circulation - expected) <= 64 * eps * 1.3 * p / (rho * speed));
}

TEST_CASE("a point vortex reaches the flow and continues bit for bit", "[euler]") {
  // Steady flow at zero incidence over the slip wall zmin, two cell zones at different states;
  // every other side is farfield with the vortex of zmin's lift below the wall.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](int iterations) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, precision);
      for (auto& b : d["boundaries"]) {
        if (b["zone"] == "zmin") {
          b["kind"] = "slip";
        } else {
          b["kind"] = "farfield";
          b["point_vortex"] = {{"zones", {"zmin"}}, {"center", {0.5, 0.0, -0.3}}};
        }
      }
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["monitors"][0]["zones"] = {"zmin"};
      d["time"]["max_iterations"] = iterations;
      d["time"]["residual_orders"] = 30.0;
    });
  };
  const auto whole = run(doc(8));
  INFO(whole.err);
  REQUIRE(whole.status == ExitStatus::finished);
  CHECK(field(whole.run, "VortexCirculation0", "RestartState").front() != 0.0);
  auto plain = doc(8);
  for (auto& b : plain["boundaries"]) b["point_vortex"] = nullptr;
  const auto without = run(plain);
  REQUIRE(without.status == ExitStatus::finished);
  CHECK(field(without.run, "VelocityZ") != field(whole.run, "VelocityZ"));

  const auto first = run(doc(4));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(8);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  CHECK(read(first.run / "history.jsonl") + read(second.run / "history.jsonl") ==
        read(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityZ", "Pressure"}) {
    CHECK(field(second.run, name) == field(whole.run, name));
  }
  CHECK(field(second.run, "VortexCirculation0", "RestartState") ==
        field(whole.run, "VortexCirculation0", "RestartState"));
}

#ifdef AUTAN_PHYSICS_CUDA
namespace {

void on_device(Json& d) { d["devices"] = {{"kind", "cuda"}, {"count", 1}}; }

}  // namespace

TEST_CASE("on a device, a uniform flow along slip walls and farfield faces stays uniform",
          "[euler][gpu]") {
  const char* precision = GENERATE("fp64", "fp32+acc64");
  const bool steady = GENERATE(true, false);
  INFO(precision << (steady ? " steady" : " unsteady"));
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, precision);
    on_device(d);
    if (steady) {
      d["time"]["max_iterations"] = 20;
    } else {
      d["time"] = {
          {"kind", "unsteady"}, {"steps", 20}, {"dt", nullptr}, {"cfl", 0.8}, {"average_from", 10}};
    }
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(out.status == ExitStatus::finished);
  const auto& f = doc["freestream"];
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const double rho = f["density"].get<double>(), p = f["pressure"].get<double>();
  const double speed = f["speed"].get<double>();
  for (const double v : field(out.run, "Density")) CHECK(std::abs(v - rho) <= 64 * eps * rho);
  for (const double v : field(out.run, "Pressure")) CHECK(std::abs(v - p) <= 64 * eps * p);
  for (const double v : field(out.run, "VelocityX")) CHECK(std::abs(v - speed) <= 64 * eps * speed);
  for (const double v : field(out.run, "VelocityZ")) CHECK(std::abs(v) <= 64 * eps * speed);
}

TEST_CASE("on a device, a verification case converges to the host's discrete solution",
          "[euler][gpu]") {
  // The device steps explicitly (local time steps), the host implicitly: both converge to the same
  // discrete solution. fp64: they differ by at most 1e-3 times its discretisation error, per field
  // in the volume-weighted L2 norm. fp32+acc64: they agree as fp32+acc64 agrees with fp64
  // (docs/validation/acceptance.md, FP32 conventions: L2 difference over the field's scale, the
  // RMS speed for velocity components, at most 500 x 2^-23).
  const std::string precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](bool device) {
    return input("hybrid-verification.solver-input.json", [&](Json& d) {
      d["precision"] = precision;
      d["time"]["max_iterations"] = device ? 200000 : 20000;
      d["time"]["residual_orders"] = precision == "fp64" ? 10.0 : 5.0;
      if (device) on_device(d);
    });
  };
  const auto host = run(doc(false));
  const auto dev = run(doc(true));
  INFO(dev.err);
  REQUIRE(host.status == ExitStatus::finished);
  REQUIRE(dev.status == ExitStatus::finished);
  CHECK(Json::parse(read(dev.run / "run.json"))["status"] == "converged");
  const auto mesh = autan::io::read_mesh(mesh_path);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<std::array<double, 3>> points(n);
  double volume = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    points[c] = {g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]};
    volume += g.cell_volume[c];
  }
  const autan::physics::Manufactured mms(read(mms_path));
  const std::array<std::string, 5> variables = {"rho", "u", "v", "w", "p"};
  const auto exact = mms.evaluate("fields", variables, points);
  auto l2 = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t c = 0; c < n; ++c) s += g.cell_volume[c] * std::pow(a[c] - b[c], 2);
    return std::sqrt(s / volume);
  };
  const std::array<const char*, 5> names = {"Density", "VelocityX", "VelocityY", "VelocityZ",
                                            "Pressure"};
  double speed = 0.0;
  for (std::size_t k = 1; k < 4; ++k) {
    const auto v = field(host.run, names[k]);
    speed += std::pow(l2(v, std::vector<double>(n, 0.0)), 2);
  }
  speed = std::sqrt(speed);
  for (std::size_t k = 0; k < 5; ++k) {
    INFO(names[k]);
    const auto a = field(host.run, names[k]), b = field(dev.run, names[k]);
    const double diff = l2(a, b);
    if (precision == "fp64") {
      const double error = l2(a, exact[k]);
      INFO("difference " << diff / error << " of the discretisation error");
      CHECK(diff <= 1e-3 * error);
    } else {
      const double scale = k >= 1 && k <= 3 ? speed : l2(a, std::vector<double>(n, 0.0));
      INFO("difference " << diff / scale / std::numeric_limits<float>::epsilon() << " eps32");
      CHECK(diff <= 500.0 * std::numeric_limits<float>::epsilon() * scale);
    }
  }
}

TEST_CASE("on a device, a run repeats bit for bit and continues bit for bit", "[euler][gpu]") {
  // Deterministic face loops (store-gather) and exact norms: the same input gives the same bits,
  // and a continued run the uninterrupted one's.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](int steps) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      inviscid_along_x(d, precision);
      on_device(d);
      for (auto& b : d["boundaries"]) b["kind"] = "farfield";
      auto& init = d["cell_zones"].back()["initial"];
      init["density"] = 1.2 * init["density"].get<double>();
      init["pressure"] = 1.3 * init["pressure"].get<double>();
      d["time"] = {{"kind", "unsteady"},
                   {"steps", steps},
                   {"dt", nullptr},
                   {"cfl", 0.8},
                   {"average_from", 2}};
    });
  };
  const auto whole = run(doc(12));
  INFO(whole.err);
  REQUIRE(whole.status == ExitStatus::finished);
  const auto again = run(doc(12));
  REQUIRE(again.status == ExitStatus::finished);
  CHECK(read(again.run / "history.jsonl") == read(whole.run / "history.jsonl"));
  const auto first = run(doc(6));
  REQUIRE(first.status == ExitStatus::finished);
  auto next = doc(12);
  next["start"] = continue_from(first.run);
  const auto second = run(next);
  INFO(second.err);
  REQUIRE(second.status == ExitStatus::finished);
  CHECK(read(first.run / "history.jsonl") + read(second.run / "history.jsonl") ==
        read(whole.run / "history.jsonl"));
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    CHECK(field(again.run, name) == field(whole.run, name));
    CHECK(field(second.run, name) == field(whole.run, name));
    CHECK(field(second.run, name, "FlowSolutionMean") ==
          field(whole.run, name, "FlowSolutionMean"));
  }
}

TEST_CASE("on a device, a flow through periodic pairs follows the host's", "[euler][gpu]") {
  // Unsteady, through both pairs and into the slip wall zmin: the device fills the periodic
  // images through the runtime's device ghost exchange, as the host through its own. After 12
  // steps the fields agree to round-off: the largest difference over the field's largest
  // magnitude (the largest speed for velocity components) within 256 eps of the precision, the
  // bound of the multi-rank test, as both sum the same fluxes in different orders.
  const char* precision = GENERATE("fp64", "fp32+acc64");
  INFO(precision);
  auto doc = [&](bool device) {
    return input("hybrid.solver-input.json", [&](Json& d) {
      periodic_box(d, precision);
      unsteady_through_pairs(d, 12);
      const double speed = d["freestream"]["speed"].get<double>();
      d["cell_zones"][0]["initial"]["velocity"] = {0.6 * speed, 0.7 * speed, -0.3 * speed};
      if (device) on_device(d);
    });
  };
  const auto host = run(doc(false));
  const auto dev = run(doc(true));
  INFO(dev.err);
  REQUIRE(host.status == ExitStatus::finished);
  REQUIRE(dev.status == ExitStatus::finished);
  const double eps = std::string(precision) == "fp64" ? std::numeric_limits<double>::epsilon()
                                                      : std::numeric_limits<float>::epsilon();
  const auto vx = field(host.run, "VelocityX"), vy = field(host.run, "VelocityY"),
             vz = field(host.run, "VelocityZ");
  double speed = 0.0;
  for (std::size_t c = 0; c < vx.size(); ++c)
    speed = std::max(speed, std::hypot(vx[c], vy[c], vz[c]));
  double worst = 0.0;
  for (const char* name : {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}) {
    const auto a = field(host.run, name), b = field(dev.run, name);
    REQUIRE(a.size() == b.size());
    double diff = 0.0, scale = 0.0;
    for (std::size_t c = 0; c < a.size(); ++c) {
      diff = std::max(diff, std::abs(a[c] - b[c]));
      scale = std::max(scale, std::abs(a[c]));
    }
    if (std::string(name).starts_with("Velocity")) scale = speed;
    worst = std::max(worst, diff / (scale * eps));
  }
  INFO("fields differ by " << worst << " eps");
  CHECK(worst <= 256.0);
}

TEST_CASE("on a device, a point vortex is refused before any directory exists", "[euler][gpu]") {
  // The device steps take no circulation from the lift (the host's steady step measures it), so a
  // point vortex on a device would keep its initial circulation: E0706, not a silent mismatch.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp64");
    on_device(d);
    for (auto& b : d["boundaries"]) {
      if (b["zone"] == "zmin") {
        b["kind"] = "slip";
      } else {
        b["kind"] = "farfield";
        b["point_vortex"] = {{"zones", {"zmin"}}, {"center", {0.5, 0.0, -0.3}}};
      }
    }
    d["monitors"][0]["zones"] = {"zmin"};
  });
  const auto out = run(doc);
  INFO(out.err);
  CHECK(out.status == ExitStatus::invalid_input);
  CHECK(!fs::exists(out.runs));
  const auto diagnostic = Json::parse(out.err.substr(0, out.err.find('\n')));
  CHECK(diagnostic["code"] == "E0706");
  CHECK(diagnostic["location"]["path"].get<std::string>().ends_with(".point_vortex"));
}

TEST_CASE("on a device, a diverged run names the cell whose stored state is not physical",
          "[euler][gpu]") {
  // As on the host: zone `lower` at 1e-9 times the freestream density rounds to zero density in
  // fp32+acc64; D0107 names a cell of that zone.
  const auto doc = input("hybrid.solver-input.json", [&](Json& d) {
    inviscid_along_x(d, "fp32+acc64");
    on_device(d);
    for (auto& z : d["cell_zones"]) {
      if (z["zone"] == "lower")
        z["initial"]["density"] = 1e-9 * d["freestream"]["density"].get<double>();
    }
    d["time"]["max_iterations"] = 5;
  });
  const auto out = run(doc);
  INFO(out.err);
  REQUIRE(!out.run.empty());
  const Json info = Json::parse(read(out.run / "run.json"));
  CHECK(info["status"] == "diverged");
  const auto& diagnostics = info["diagnostics"];
  const auto d107 = std::find_if(diagnostics.begin(), diagnostics.end(),
                                 [](const Json& g) { return g["code"] == "D0107"; });
  REQUIRE(d107 != diagnostics.end());
  CHECK((*d107)["data"]["zone"] == "lower");
}
#endif
