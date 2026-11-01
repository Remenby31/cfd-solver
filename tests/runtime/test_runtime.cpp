#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cgnslib.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "autan/core/geometry.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/partition/partition.hpp"
#include "autan/runtime/input.hpp"
#include "autan/runtime/run.hpp"
#include "relaxation.hpp"

extern char** environ;

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using autan::runtime::ExitStatus;
using Catch::Matchers::WithinRel;

const fs::path data_dir{AUTAN_TEST_DATA_DIR};
const fs::path mesh_path = data_dir / "mesh" / "hybrid" / "polyMesh";
// The Euler manufactured solution autan-vv exported for the SDK's tests.
const fs::path mms_path = data_dir / ".." / ".." / "py" / "tests" / "fixtures" / "euler.mms.json";
const autan::runtime::SolverBuild build{.version = "0.0.0", .commit = std::string(40, 'a')};

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

fs::path fresh_dir() {
  std::random_device random;
  const fs::path dir = fs::temp_directory_path() / ("autan_runtime-" + std::to_string(random()));
  fs::create_directories(dir);
  return dir;
}

// The solver input `autan prepare` wrote for tests/data/mesh/hybrid (steady
// SA, wall ymin, symmetry zmax, farfield elsewhere, forces on ymin every 5),
// pointed at this checkout's mesh and edited by `edit`.
Json input(const std::function<void(Json&)>& edit = {},
           const char* file = "hybrid.solver-input.json") {
  Json doc = Json::parse(read(data_dir / "runtime" / file));
  doc["mesh"]["path"] = fs::absolute(mesh_path).string();
  if (doc["verification"].is_object())
    doc["verification"]["path"] = fs::absolute(mms_path).string();
  if (edit) edit(doc);
  return doc;
}

struct Outcome {
  ExitStatus status;
  fs::path runs;
  fs::path run;  // empty when no directory was created
  std::string err;
};

Outcome run(const Json& doc, const autan::runtime::PhysicsRegistry& registry,
            const fs::path& stop_file = {}) {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << doc.dump();
  std::FILE* out = std::tmpfile();
  std::FILE* err = std::tmpfile();
  const ExitStatus status =
      autan::runtime::run_case(input_file, dir / "runs", registry, build, out, err, stop_file);
  auto text = [](std::FILE* f) {
    std::rewind(f);
    std::string s;
    for (int c = std::fgetc(f); c != EOF; c = std::fgetc(f)) s.push_back(static_cast<char>(c));
    std::fclose(f);
    return s;
  };
  std::string printed = text(out);
  if (!printed.empty() && printed.back() == '\n') printed.pop_back();
  return Outcome{.status = status, .runs = dir / "runs", .run = printed, .err = text(err)};
}

Outcome run(const Json& doc, autan_test::RelaxationOptions options = {},
            const fs::path& stop_file = {}) {
  return run(doc, autan_test::relaxation_registry(options), stop_file);
}

// The hybrid input on periodic-box (Gmsh): xmin -> xmax by (tx, 0, 0), ymin
// -> ymax by (0, 1, 0), zmin a wall, zmax a symmetry plane.
Json periodic_input(double tx, const std::function<void(Json&)>& edit = {}) {
  const fs::path box = data_dir / "mesh" / "periodic-box" / "periodic-box.ascii.msh";
  return input([&](Json& d) {
    d["mesh"]["path"] = fs::absolute(box).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(box);
    d["periodic"] =
        Json::array({Json{{"zones", {"xmin", "xmax"}},
                          {"transform", {{"kind", "translation"}, {"vector", {tx, 0.0, 0.0}}}}},
                     Json{{"zones", {"ymin", "ymax"}},
                          {"transform", {{"kind", "translation"}, {"vector", {0.0, 1.0, 0.0}}}}}});
    // zmin takes the wall condition of ymin (now periodic), zmax keeps its own.
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
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    d["cell_zones"] = Json::array({zone});
    d["monitors"][0]["zones"] = Json::array({"zmin"});
    if (edit) edit(d);
  });
}

std::vector<Json> history(const fs::path& run) {
  std::vector<Json> lines;
  std::ifstream in(run / "history.jsonl");
  for (std::string line; std::getline(in, line);) lines.push_back(Json::parse(line));
  return lines;
}

std::vector<double> solution_field(const fs::path& file, const char* solution, const char* name) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  int count = 0;
  REQUIRE(cg_nsols(fn, 1, 1, &count) == CG_OK);
  std::vector<double> values;
  for (int s = 1; s <= count; ++s) {
    char sol_name[33];
    CGNS_ENUMT(GridLocation_t) location{};
    REQUIRE(cg_sol_info(fn, 1, 1, s, sol_name, &location) == CG_OK);
    if (std::string(sol_name) != solution) continue;
    char zone_name[33];
    cgsize_t size[9];
    REQUIRE(cg_zone_read(fn, 1, 1, zone_name, size) == CG_OK);
    values.resize(static_cast<std::size_t>(size[1]));
    const cgsize_t first = 1;
    REQUIRE(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &size[1],
                          values.data()) == CG_OK);
  }
  cg_close(fn);
  return values;
}

std::vector<std::string> field_list(const fs::path& file) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  int count = 0;
  REQUIRE(cg_nfields(fn, 1, 1, 1, &count) == CG_OK);
  std::vector<std::string> names;
  for (int f = 1; f <= count; ++f) {
    char name[33];
    CGNS_ENUMT(DataType_t) type{};
    REQUIRE(cg_field_info(fn, 1, 1, 1, f, &type, name) == CG_OK);
    names.emplace_back(name);
  }
  cg_close(fn);
  return names;
}

void require_error(const Outcome& o, const std::string& code) {
  CHECK(o.status == ExitStatus::invalid_input);
  CHECK(o.run.empty());
  CHECK((!fs::exists(o.runs) || fs::is_empty(o.runs)));
  REQUIRE(!o.err.empty());
  CHECK(Json::parse(o.err)["code"] == code);
}

std::string file_sha256(const fs::path& file) {
  const std::string bytes = read(file);
  std::string hash;
  picosha2::hash256_hex_string(bytes, hash);
  return hash;
}

// An autan.initial-state/1 document for the fluid cells of `mesh_file`
// (every cell of the hybrid box), rows in reverse cell order: density
// 1 + 1e-3 c, velocity (c, -c, 0.5), pressure 1e5 + c for cell c.
Json initial_state_document(const fs::path& mesh_file) {
  const autan::Mesh mesh = autan::io::read_mesh(mesh_file);
  const autan::Geometry g = autan::compute_geometry(mesh);
  Json centroids = Json::array(), values = Json::array();
  for (autan::index_t c = mesh.n_cells() - 1; c >= 0; --c) {
    const auto i = static_cast<std::size_t>(c);
    const auto x = static_cast<double>(c);
    centroids.push_back({g.cell_centroid_x[i], g.cell_centroid_y[i], g.cell_centroid_z[i]});
    values.push_back({1.0 + 1e-3 * x, x, -x, 0.5, 1e5 + x});
  }
  return Json{{"schema", "autan.initial-state/1"},
              {"oracle", "test"},
              {"mesh_sha256", autan::io::mesh_hash(mesh_file)},
              {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
              {"cells", mesh.n_cells()},
              {"centroids", centroids},
              {"values", values}};
}

// The hybrid input made inviscid: an initial-state file has no turbulence
// variables (initial-state.md).
void inviscid(Json& d) {
  d["physics"]["turbulence"]["model"] = "inviscid";
  d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
  d["freestream"]["nu_tilde"] = nullptr;
  for (auto& zone : d["cell_zones"]) zone["initial"]["nu_tilde"] = nullptr;
}

// `doc` written to a fresh file; the solver input's initial_state names it.
Json with_initial_state(Json input_doc, const Json& doc) {
  const fs::path file = fresh_dir() / "initial.json";
  std::ofstream(file) << doc.dump();
  input_doc["initial_state"] = {{"path", file.string()}, {"sha256", file_sha256(file)}};
  return input_doc;
}

// The `start` the SDK writes for a run that starts from `previous`
// (solver-input.md): its run.json, the hash of its solution and, for a
// steady `continue`, the residuals at iteration 1 of the chain's first run
// and the equations exempt from them that went above their floor in the
// chain (case-schema.md#time).
Json start_from(const fs::path& previous, const std::string& mode) {
  const Json info = Json::parse(read(previous / "run.json"));
  Json reference = nullptr;
  Json woken = nullptr;
  if (mode == "continue" && info["time"].is_null()) {
    const bool chained = !info["start"].is_null() && info["start"]["mode"] == "continue";
    reference = chained ? info["start"]["residual_reference"] : history(previous)[0]["residuals"];
    double largest = 0.0;
    for (const auto& [name, f] : reference.items()) largest = std::max(largest, f.get<double>());
    const double floor = 1e-12 * largest;
    std::set<std::string> names;
    if (chained) {
      for (const auto& n : info["start"]["woken"]) names.insert(n.get<std::string>());
    }
    for (const auto& line : history(previous)) {
      for (const auto& [name, f] : reference.items()) {
        if (f.get<double>() < floor && line["residuals"][name].get<double>() > floor)
          names.insert(name);
      }
    }
    woken = Json(std::vector<std::string>(names.begin(), names.end()));
  }
  return Json{{"run", previous.string()},
              {"run_id", info["run_id"]},
              {"identity", info["identity"]},
              {"mode", mode},
              {"iterations", info["iterations"]},
              {"time", info["time"]},
              {"solution_sha256", file_sha256(previous / "solution.cgns")},
              {"residual_reference", reference},
              {"woken", woken}};
}

std::vector<std::string> lines_of(const fs::path& file) {
  std::vector<std::string> lines;
  std::ifstream in(file);
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

}  // namespace

TEST_CASE("A converging run writes the run directory of results.md", "[runtime]") {
  // Residuals 0.5^(k-1) drop 8 orders at k = 28 (0.5^27 < 1e-8 < 0.5^26).
  const Json doc = input();
  const Outcome o = run(doc);
  REQUIRE(o.status == ExitStatus::finished);
  REQUIRE(fs::is_directory(o.run));
  for (const auto& entry : fs::directory_iterator(o.runs)) {
    CHECK(entry.path().filename().string()[0] != '.');  // the staging directory is gone
  }
  std::vector<std::string> files;
  for (const auto& entry : fs::directory_iterator(o.run))
    files.push_back(entry.path().filename().string());
  std::ranges::sort(files);
  CHECK(files == std::vector<std::string>{"case.toml", "history.jsonl", "log.jsonl",
                                          "mesh-info.json", "run.json", "solution.cgns"});

  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["schema"] == "autan.results/1");
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);
  // results.md, timing: this run's 28 iterations, the loop within the
  // process, and ns per cell and iteration by its definition.
  const Json& t = info["timing"];
  REQUIRE(t.is_object());
  CHECK(t["iterations"] == 28);
  CHECK(t["solve_seconds"].get<double>() > 0.0);
  CHECK(t["solve_seconds"].get<double>() <= t["wall_seconds"].get<double>());
  const double cells = static_cast<double>(autan::io::read_mesh(mesh_path).n_cells());
  CHECK_THAT(
      t["ns_per_cell_iteration"].get<double>(),
      Catch::Matchers::WithinRel(t["solve_seconds"].get<double>() * 1e9 / (cells * 28), 1e-12));
  CHECK(std::regex_match(info["run_id"].get<std::string>(),
                         std::regex("run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")));
  CHECK(o.run.filename() == info["run_id"].get<std::string>());
  CHECK(std::regex_match(info["finished"].get<std::string>(),
                         std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z")));
  CHECK(info["mesh"]["path"] == fs::absolute(mesh_path).lexically_normal().string());
  CHECK(info["solver"] == Json{{"version", "0.0.0"}, {"commit", build.commit}, {"backend", "cpu"}});
  // identity: SHA-256 of the seven fields joined by "\n" (results.md).
  std::string expected;
  picosha2::hash256_hex_string(doc["case"]["sha256"].get<std::string>() + "\n" +
                                   doc["mesh"]["sha256"].get<std::string>() + "\n" + build.commit +
                                   "\nfp32+acc64\ncpu\ncpu\n1",
                               expected);
  CHECK(info["identity"] == expected);
  CHECK(read(o.run / "case.toml") == doc["case"]["toml"].get<std::string>());
  CHECK(read(o.run / "mesh-info.json") ==
        autan::io::mesh_info_json(autan::io::read_mesh(mesh_path),
                                  doc["mesh"]["sha256"].get<std::string>()));

  const auto lines = history(o.run);
  REQUIRE(lines.size() == 28);
  const std::vector<std::string> equations{"density",    "momentum-x", "momentum-y",
                                           "momentum-z", "energy",     "nu-tilde"};
  for (std::size_t k = 0; k < lines.size(); ++k) {
    CHECK(lines[k]["iteration"] == k + 1);
    CHECK(lines[k]["time"].is_null());
    for (std::size_t e = 0; e < equations.size(); ++e) {
      CHECK(lines[k]["residuals"][equations[e]] ==
            static_cast<double>(e + 1) * std::pow(0.5, static_cast<double>(k)));
    }
    // Forces on ymin every 5 iterations; uniform p = p_inf gives no load.
    const bool recorded = (k + 1) % 5 == 0;
    CHECK(lines[k]["monitors"].contains("forces") == recorded);
    if (recorded) {
      const Json& f = lines[k]["monitors"]["forces"];
      CHECK(f.size() == 12);  // fx..mz, cd cs cl (ref_area), cmx..cmz (ref_length)
      for (const auto& [key, value] : f.items()) CHECK(value == 0.0);
    }
  }
  // Convergence between two monitor iterations: run.json still holds the
  // final forces, at the last iteration.
  const Json& final_forces = info["monitors"]["forces"];
  CHECK(final_forces.size() == 13);
  CHECK(final_forces["iteration"] == 28);
  for (const auto& [key, value] : final_forces.items()) {
    if (key != "iteration") CHECK(value == 0.0);
  }

  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  CHECK(back.n_cells() == mesh.n_cells());
  const auto density = solution_field(o.run / "solution.cgns", "FlowSolution", "Density");
  REQUIRE(density.size() == static_cast<std::size_t>(mesh.n_cells()));
  for (const double rho : density)
    CHECK_THAT(rho, WithinRel(doc["freestream"]["density"].get<double>(), 1e-15));
}

TEST_CASE("Every final status is recorded as results.md defines it", "[runtime]") {
  SECTION("completed: budget reached before convergence") {
    const Outcome o =
        run(input([](Json& d) { d["time"]["max_iterations"] = 10; }), {.factor = 0.9});
    CHECK(o.status == ExitStatus::finished);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "completed");
    CHECK(info["iterations"] == 10);
    CHECK(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("final monitors: the last state, equal to its history line when it has one") {
    // The pressure grows at each step, so each line's forces differ.
    const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 10; }),
                          {.factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3});
    const Json info = Json::parse(read(o.run / "run.json"));
    Json expected = history(o.run)[9]["monitors"]["forces"];
    CHECK(expected != history(o.run)[4]["monitors"]["forces"]);
    expected["iteration"] = 10;
    CHECK(info["monitors"] == Json{{"forces", expected}});

    // Between two monitor iterations: ymin, the unit square y = 0 with
    // outward normal -y, carries fy = -(p - p_inf), p = (p_inf + dp) 1.001^11.
    const Json doc = input([](Json& d) { d["time"]["max_iterations"] = 11; });
    const Outcome off = run(doc, {.factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3});
    const Json f = Json::parse(read(off.run / "run.json"))["monitors"]["forces"];
    CHECK(f["iteration"] == 11);
    const double p_inf = doc["freestream"]["pressure"].get<double>();
    CHECK_THAT(f["fy"].get<double>(),
               WithinRel(-((p_inf + 1000.0) * std::pow(1.001, 11) - p_inf), 1e-12));
  }
  SECTION("stopped before iteration 1: the initial state at iteration 0") {
    // ymin, the unit square y = 0 with outward normal -y, under a pressure
    // excess dp carries F = (0, -dp, 0).
    const Outcome o =
        run(input(), {.factor = 0.9, .pressure_offset = 1000.0, .signal_on_start = true});
    CHECK(o.status == ExitStatus::signalled);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "stopped");
    CHECK(info["iterations"] == 0);
    CHECK(info["timing"].is_null());  // no iteration to time
    CHECK_FALSE(fs::exists(o.run / "history.jsonl"));
    const Json& f = info["monitors"]["forces"];
    CHECK(f["iteration"] == 0);
    CHECK_THAT(f["fy"].get<double>(), Catch::Matchers::WithinAbs(-1000.0, 1e-9));
    CHECK(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("failed in the final loads: E0704, no monitors") {
    const Outcome o = run(input([](Json& d) { d["monitors"][0]["every"] = 1000; }),
                          {.factor = 0.5, .throw_in_loads = true});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["iterations"] == 28);
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
  }
  SECTION("failed on final loads that overflow: E0704, no monitors") {
    // A pressure excess on ymin about a centre 1e308 m away: moments beyond
    // the double range.
    const Outcome o = run(input([](Json& d) {
                            d["monitors"][0]["every"] = 1000;
                            d["monitors"][0]["moment_center"] = {1e308, 1e308, 1e308};
                          }),
                          {.factor = 0.5, .pressure_offset = 1000.0});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
  }
  SECTION("failed on a recorded load that is not finite: E0704, history up to the line before") {
    // results.md, History: a record holds numbers, so a non-finite force
    // at iteration 5 fails the run there instead of writing null.
    const Outcome o = run(input([](Json& d) { d["monitors"][0]["every"] = 1; }),
                          {.factor = 0.9, .nan_loads_from = 5});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["iterations"] == 4);
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
    const auto lines = history(o.run);
    REQUIRE(lines.size() == 4);
    CHECK(lines.back()["iteration"] == 4);
    CHECK(std::isfinite(lines.back()["monitors"]["forces"]["fy"].get<double>()));
  }
  SECTION("failed writing the solution: E0704, no monitors") {
    const Outcome o = run(input(), {.factor = 0.5, .throw_in_fields = true});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(info["monitors"].is_null());
    CHECK(info["timing"].is_null());
  }
  SECTION("diverged: D0107 with the iteration, cell and zone") {
    const Outcome o = run(input(), {.factor = 0.9, .diverge_at = 7, .diverge_cell = 3});
    CHECK(o.status == ExitStatus::finished);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "diverged");
    CHECK(info["iterations"] == 6);
    REQUIRE(info["diagnostics"].size() == 1);
    const Json& d = info["diagnostics"][0];
    CHECK(d["code"] == "D0107");
    CHECK(d["severity"] == "divergence");
    CHECK(d["data"]["iteration"] == 7);
    CHECK(d["data"]["cell"] == 3);
    const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
    CHECK(d["data"]["zone"] ==
          mesh.cell_zones()[static_cast<std::size_t>(mesh.cell_zone()[3])].name);
    CHECK(history(o.run).size() == 6);
    CHECK(info["monitors"].is_null());
    // The diverged state holds a NaN: no solution can satisfy the readers.
    CHECK_FALSE(fs::exists(o.run / "solution.cgns"));
  }
  SECTION("failed: E0704, history kept, exit 1") {
    const Outcome o = run(input(), {.factor = 0.9, .throw_at = 4});
    CHECK(o.status == ExitStatus::failed);
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "failed");
    CHECK(info["diagnostics"][0]["code"] == "E0704");
    CHECK(history(o.run).size() == 3);
    CHECK(info["monitors"].is_null());
    CHECK(info["timing"].is_null());  // a failed run is not timed
  }
  SECTION("unsteady with averaging: time, averaging and FlowSolutionMean") {
    // Written by `autan prepare` with time = Unsteady(steps=6, dt=0.01 s,
    // average_from=3); fields grow by 10 % per step, so the mean is
    // (1.1^4 + 1.1^5 + 1.1^6) / 3 times the initial value.
    const Outcome o = run(input({}, "hybrid-unsteady.solver-input.json"), {.growth = 0.1});
    const Json info = Json::parse(read(o.run / "run.json"));
    CHECK(info["status"] == "completed");
    CHECK_THAT(info["time"].get<double>(), WithinRel(0.06, 1e-12));
    CHECK(info["averaging"]["from_step"] == 4);
    CHECK(info["averaging"]["to_step"] == 6);
    CHECK_THAT(info["averaging"]["from_time"].get<double>(), WithinRel(0.03, 1e-12));
    CHECK_THAT(info["averaging"]["to_time"].get<double>(), WithinRel(0.06, 1e-12));
    const auto lines = history(o.run);
    REQUIRE(lines.size() == 6);
    CHECK_THAT(lines[1]["time"].get<double>(), WithinRel(0.02, 1e-12));
    const auto mean = solution_field(o.run / "solution.cgns", "FlowSolutionMean", "Density");
    const auto last = solution_field(o.run / "solution.cgns", "FlowSolution", "Density");
    const double rho =
        input({}, "hybrid-unsteady.solver-input.json")["freestream"]["density"].get<double>();
    REQUIRE(mean.size() == last.size());
    for (std::size_t c = 0; c < mean.size(); ++c) {
      CHECK_THAT(
          mean[c],
          WithinRel(rho * (std::pow(1.1, 4) + std::pow(1.1, 5) + std::pow(1.1, 6)) / 3.0, 1e-12));
      CHECK_THAT(last[c], WithinRel(rho * std::pow(1.1, 6), 1e-12));
    }
  }
}

TEST_CASE("Invalid inputs exit 2 with a coded diagnostic and no directory", "[runtime]") {
  SECTION("E0705 unknown key") {
    require_error(run(input([](Json& d) { d["physics"]["extra"] = 1; })), "E0705");
  }
  SECTION("E0705 missing key") {
    require_error(run(input([](Json& d) { d["freestream"].erase("speed"); })), "E0705");
  }
  SECTION("E0705 value a kind forbids") {
    require_error(run(input([](Json& d) { d["boundaries"][0]["static_pressure"] = 1e5; })),
                  "E0705");
  }
  SECTION("E0705 states on a condition that takes none") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["states"] = {{"path", "/s.json"},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 oracle-state without an absolute states path") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["kind"] = "oracle-state";
                    d["boundaries"][0]["states"] = {{"path", "s.json"},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 oracle-state with a turbulence model") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["kind"] = "oracle-state";
                    d["boundaries"][0]["states"] = {{"path", "/s.json"},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 NUL in a states path") {
    require_error(run(input([](Json& d) {
                    d["physics"]["turbulence"]["model"] = "inviscid";
                    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
                    d["freestream"]["nu_tilde"] = nullptr;
                    for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
                    d["boundaries"][0]["kind"] = "oracle-state";
                    d["boundaries"][0]["states"] = {{"path", std::string("/s\0.json", 8)},
                                                    {"sha256", std::string(64, 'a')}};
                  })),
                  "E0705");
  }
  SECTION("E0705 point vortex on a condition other than farfield") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][2]["point_vortex"] = {{"zones", {"ymin"}},
                                                          {"center", {0.0, 0.0, 0.0}}};
                  })),
                  "E0705");
  }
  SECTION("E0705 point vortex over a zone that is not a wall") {
    require_error(run(input([](Json& d) {
                    d["boundaries"][0]["point_vortex"] = {{"zones", {"xmax"}},
                                                          {"center", {0.0, 0.0, 0.0}}};
                  })),
                  "E0705");
  }
  SECTION("E0705 zone absent from the mesh") {
    require_error(run(input([](Json& d) { d["boundaries"][0]["zone"] = "nowhere"; })), "E0705");
  }
  SECTION("E0203 mesh hash") {
    require_error(run(input([](Json& d) { d["mesh"]["sha256"] = std::string(64, '0'); })), "E0203");
  }
  SECTION("E0706 no physics in the build") {
    require_error(run(input(), autan::runtime::PhysicsRegistry{}), "E0706");
  }
  SECTION("E0146 a launcher's ranks without MPI started") {
    // As a launcher would start one of four ranks, without MPI initialized.
    REQUIRE(setenv("OMPI_COMM_WORLD_SIZE", "4", 1) == 0);
    const Outcome o = run(input());
    unsetenv("OMPI_COMM_WORLD_SIZE");
    require_error(o, "E0146");
    // PMIx may give the rank alone: without a count, even rank 0 cannot
    // tell it runs alone.
    REQUIRE(setenv("PMIX_RANK", "0", 1) == 0);
    const Outcome pmix = run(input());
    unsetenv("PMIX_RANK");
    require_error(pmix, "E0146");
  }
}

TEST_CASE("A device run needs a build and a device that drive it", "[runtime]") {
  // Never a silent CPU run: a kind this build does not drive is E0706, and
  // a CUDA build that sees no device is E0807, both before any directory,
  // whatever the physics registered: here, one for the CPU only.
  autan::runtime::PhysicsRegistry cpu_only;
  cpu_only.add({.kind = "*", .turbulence = "*", .precision = "*", .device = "cpu"},
               [](const autan::runtime::PhysicsContext& context) {
                 return std::make_unique<autan_test::Relaxation>(context,
                                                                 autan_test::RelaxationOptions{});
               });
  const Outcome hip = run(input([](Json& d) { d["devices"]["kind"] = "hip"; }), cpu_only);
  require_error(hip, "E0706");
  CHECK(Json::parse(hip.err)["data"]["path"] == "devices.kind");
#ifdef AUTAN_HAVE_CUDA
  // As a host without a GPU: CUDA reads this before its first call, and
  // this test runs in a process of its own.
  REQUIRE(setenv("CUDA_VISIBLE_DEVICES", "-1", 1) == 0);
  const Outcome o = run(input([](Json& d) { d["devices"]["kind"] = "cuda"; }), cpu_only);
  require_error(o, "E0807");
  CHECK(Json::parse(o.err)["data"] ==
        Json{{"visible", 0}, {"needed", 1}, {"path", "devices.kind"}});
#else
  const Outcome o = run(input([](Json& d) { d["devices"]["kind"] = "cuda"; }), cpu_only);
  require_error(o, "E0706");
  CHECK(Json::parse(o.err)["data"]["path"] == "devices.kind");
#endif
}

#ifdef AUTAN_HAVE_CUDA
TEST_CASE("A CUDA run gives the physics its device and stream", "[runtime][gpu]") {
  // On a periodic mesh too, where the relaxation physics checks that the
  // device exchange fills each image ghost on the run's stream.
  const bool periodic = GENERATE(false, true);
  CAPTURE(periodic);
  const auto on_cuda = [](Json& d) { d["devices"]["kind"] = "cuda"; };
  const Outcome o = run(periodic ? periodic_input(1.0, on_cuda) : input(on_cuda));
  REQUIRE(o.status == ExitStatus::finished);
  std::ifstream log_in(o.run / "log.jsonl");
  Json ready, cuda;
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    if (event["message"] == "relaxation physics ready") ready = event["data"];
    if (event["source"] == "cuda") cuda = event["data"];
  }
  CHECK(ready["device"] == 0);
  CHECK(ready["stream"] == true);
  CHECK(cuda["device"] == 0);
  CHECK(!cuda["name"].get<std::string>().empty());
  CHECK(cuda["visible"].get<int>() >= 1);
}
#endif

TEST_CASE("A scope's refusal carries its own code", "[runtime]") {
  // E0706 unless the physics says more (E1109 for a reproducibility it cannot honour): any
  // registered code goes through, here E0705.
  autan_test::RelaxationOptions refusing;
  refusing.refuse_code = "E0705";
  const Outcome o = run(input(), refusing);
  require_error(o, "E0705");
  CHECK(Json::parse(o.err)["data"]["path"] == "physics");
}

TEST_CASE("A physics' scope refuses an input before any directory", "[runtime]") {
  const bool refuse = GENERATE(true, false);
  autan::runtime::PhysicsRegistry registry;
  registry.add(
      {.kind = "*", .turbulence = "*", .precision = "*", .device = "*"},
      [](const autan::runtime::PhysicsContext& context) {
        return std::make_unique<autan_test::Relaxation>(context, autan_test::RelaxationOptions{});
      },
      [refuse](const autan::runtime::SolverInput& in,
               int ranks) -> std::optional<autan::runtime::PhysicsRefusal> {
        // The scope sees the run's rank count: one, here.
        if (!refuse || ranks != 1) return std::nullopt;
        return autan::runtime::PhysicsRefusal{
            .message = "refused",
            .path = "cell_zones[1].medium",
            .data_json = Json{{"zone", in.cell_zones[1].zone}}.dump()};
      });
  const Json doc = input();
  const Outcome o = run(doc, registry);
  if (!refuse) {
    CHECK(o.status == ExitStatus::finished);
    return;
  }
  require_error(o, "E0706");
  const Json d = Json::parse(o.err);
  CHECK(d["location"]["path"] == "cell_zones[1].medium");
  CHECK(d["data"] ==
        Json{{"zone", doc["cell_zones"][1]["zone"]}, {"path", "cell_zones[1].medium"}});
}

TEST_CASE("Deterministic runs produce identical history and solution arrays", "[runtime]") {
  const Json doc = input([](Json& d) { d["reproducibility"] = "run"; });
  const Outcome a = run(doc);
  const Outcome b = run(doc);
  CHECK(read(a.run / "history.jsonl") == read(b.run / "history.jsonl"));
  for (const char* field : {"Density", "Pressure", "TurbulentSANuTilde"}) {
    CHECK(solution_field(a.run / "solution.cgns", "FlowSolution", field) ==
          solution_field(b.run / "solution.cgns", "FlowSolution", field));
  }
}

TEST_CASE("A stop file stops the run as SIGINT does", "[runtime]") {
  // The relaxation physics creates the file during iteration 5: the run
  // stops before iteration 6, with its solution, and exits 130.
  const fs::path stop = fresh_dir() / "stop";
  autan_test::RelaxationOptions options{.factor = 0.9};
  options.touch_at = 5;
  options.touch_file = stop.string();
  const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 1000; }), options, stop);
  CHECK(o.status == ExitStatus::signalled);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "stopped");
  CHECK(info["iterations"] == 5);
  CHECK(fs::exists(o.run / "solution.cgns"));
}

TEST_CASE("solve_seconds leaves out the physics' log writes", "[runtime]") {
  // Every step logs 10 MB: the log takes most of the run, which the solve
  // time does not count (results.md, timing).
  autan_test::RelaxationOptions options{.factor = 0.9};
  options.log_bytes = 10'000'000;
  const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 4; }), options);
  REQUIRE(o.status == ExitStatus::finished);
  const Json t = Json::parse(read(o.run / "run.json"))["timing"];
  CHECK(t["solve_seconds"].get<double>() < 0.25 * t["wall_seconds"].get<double>());
}

TEST_CASE("SIGTERM stops a running process: stopped, solution written, exit 130", "[runtime]") {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  std::ofstream(input_file) << input([](Json& d) { d["time"]["max_iterations"] = 100000; }).dump();
  const std::string runs = (dir / "runs").string();
  std::vector<std::string> args{AUTAN_RUN_RELAXATION, "--input", input_file.string(), "--runs",
                                runs};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  std::vector<std::string> env{"AUTAN_TEST_FACTOR=0.999", "AUTAN_TEST_PAUSE_MS=20"};
  for (char** e = environ; *e != nullptr; ++e) env.emplace_back(*e);
  std::vector<char*> envp;
  for (auto& e : env) envp.push_back(e.data());
  envp.push_back(nullptr);
  pid_t pid = 0;
  REQUIRE(posix_spawn(&pid, AUTAN_RUN_RELAXATION, nullptr, nullptr, argv.data(), envp.data()) == 0);

  // Wait for a few recorded iterations, then ask the process to stop.
  fs::path run_dir;
  for (int attempt = 0; attempt < 500 && run_dir.empty(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!fs::exists(runs)) continue;
    for (const auto& entry : fs::directory_iterator(runs)) {
      if (entry.path().filename().string()[0] != '.' &&
          fs::exists(entry.path() / "history.jsonl") && history(entry.path()).size() >= 3) {
        run_dir = entry.path();
      }
    }
  }
  REQUIRE(!run_dir.empty());
  REQUIRE(kill(pid, SIGTERM) == 0);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 130);
  const Json info = Json::parse(read(run_dir / "run.json"));
  CHECK(info["status"] == "stopped");
  CHECK(info["iterations"].get<int>() >= 3);
  CHECK(info["iterations"].get<std::size_t>() == history(run_dir).size());
  CHECK(fs::exists(run_dir / "solution.cgns"));
}

TEST_CASE("Forces monitors report the loads and their coefficients", "[runtime]") {
  // ymin is the unit square y = 0 with outward normal -y: a pressure excess
  // dp gives F = (0, -dp, 0) and, about the origin, M = c x F with c the
  // square's centroid (1/2, 0, 1/2). Coefficients follow results.md with the
  // drag, lift and side directions of the incidence and sideslip.
  constexpr double dp = 1000.0, alpha = 0.3, beta = 0.2;
  const Json doc = input([&](Json& d) {
    d["freestream"]["aoa"] = alpha;
    d["freestream"]["sideslip"] = beta;
  });
  const Outcome o = run(doc, {.factor = 0.5, .pressure_offset = dp});
  const Json f = history(o.run)[4]["monitors"]["forces"];
  const double q = 0.5 * doc["freestream"]["density"].get<double>() *
                   std::pow(doc["freestream"]["speed"].get<double>(), 2);
  const std::array<double, 3> force{0.0, -dp, 0.0};
  const std::array<double, 3> moment{0.5 * dp, 0.0, -0.5 * dp};
  const std::array<double, 3> drag{std::cos(alpha) * std::cos(beta), -std::sin(beta),
                                   std::sin(alpha) * std::cos(beta)};
  const std::array<double, 3> lift{-std::sin(alpha), 0.0, std::cos(alpha)};
  const std::array<double, 3> side{lift[1] * drag[2] - lift[2] * drag[1],
                                   lift[2] * drag[0] - lift[0] * drag[2],
                                   lift[0] * drag[1] - lift[1] * drag[0]};
  auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const std::array<std::string, 3> axes{"x", "y", "z"};
  // A history line, and run.json's final record at iteration 28, between two
  // monitor iterations.
  const Json final_forces = Json::parse(read(o.run / "run.json"))["monitors"]["forces"];
  CHECK(final_forces["iteration"] == 28);
  for (const Json& r : {f, final_forces}) {
    for (std::size_t k = 0; k < 3; ++k) {
      CHECK_THAT(r["f" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(force[k], 1e-9));
      CHECK_THAT(r["m" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(moment[k], 1e-9));
      CHECK_THAT(r["cm" + axes[k]].get<double>(), Catch::Matchers::WithinAbs(moment[k] / q, 1e-9));
    }
    CHECK_THAT(r["cd"].get<double>(), Catch::Matchers::WithinAbs(dot(force, drag) / q, 1e-12));
    CHECK_THAT(r["cl"].get<double>(), Catch::Matchers::WithinAbs(dot(force, lift) / q, 1e-12));
    CHECK_THAT(r["cs"].get<double>(), Catch::Matchers::WithinAbs(dot(force, side) / q, 1e-12));
  }
  CHECK(std::abs(f["cd"].get<double>()) > 1e-3);
}

TEST_CASE("An incompressible input from the SDK runs", "[runtime]") {
  // `autan prepare` for water through the hybrid box: velocity inlet,
  // outlet, walls; no temperature anywhere.
  const Outcome o = run(input({}, "hybrid-water.solver-input.json"));
  REQUIRE(o.status == ExitStatus::finished);
  const auto lines = history(o.run);
  CHECK(lines.front()["residuals"].contains("pressure"));
  CHECK_FALSE(lines.front()["residuals"].contains("energy"));
  CHECK(field_list(o.run / "solution.cgns") ==
        std::vector<std::string>{"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"});
}

TEST_CASE("The input reader enforces the solver-input rules", "[runtime]") {
  SECTION("WMLES needs its wall model") {
    require_error(run(input([](Json& d) {
                    d["physics"]["turbulence"] =
                        Json{{"model", "wmles"},      {"base", nullptr},
                             {"wall_model", nullptr}, {"nu_tilde_ratio", nullptr},
                             {"intensity", nullptr},  {"viscosity_ratio", nullptr}};
                  })),
                  "E0705");
  }
  SECTION("DDES needs an unsteady time") {
    require_error(run(input([](Json& d) {
                    d["physics"]["turbulence"]["model"] = "ddes";
                    d["physics"]["turbulence"]["base"] = "sa";
                  })),
                  "E0705");
  }
  SECTION("monitor names are unique") {
    require_error(run(input([](Json& d) { d["monitors"].push_back(d["monitors"][0]); })), "E0705");
  }
  SECTION("cell zones are described once") {
    require_error(run(input([](Json& d) { d["cell_zones"].push_back(d["cell_zones"][0]); })),
                  "E0705");
  }
  SECTION("periodic pairs are the mesh's, well formed") {
    const Json pair{{"zones", {"xmin", "xmax"}},
                    {"transform", {{"kind", "translation"}, {"vector", {1.0, 0.0, 0.0}}}}};
    // The hybrid mesh has no periodic pair.
    require_error(run(input([&](Json& d) { d["periodic"] = Json::array({pair}); })), "E0705");
    require_error(run(input([&](Json& d) {
                    Json bad = pair;
                    bad["transform"]["kind"] = "rotation";
                    d["periodic"] = Json::array({bad});
                  })),
                  "E0705");
  }
  SECTION("one rank per device: devices.count is the rank count") {
    const Outcome o = run(input([](Json& d) { d["devices"]["count"] = 2; }));
    require_error(o, "E0705");
    CHECK(Json::parse(o.err)["location"]["path"] == "devices.count");
  }
  SECTION("mesh path is absolute") {
    require_error(run(input([](Json& d) { d["mesh"]["path"] = "relative/polyMesh"; })), "E0705");
  }
  SECTION("hashes are 64 lowercase hexadecimal characters") {
    require_error(run(input([](Json& d) { d["case"]["sha256"] = "ABC"; })), "E0705");
  }
  SECTION("a number beyond double range") {
    std::string text = input().dump();
    const std::string key = "\"speed\":";
    const auto at = text.find(key) + key.size();
    text.replace(at, text.find(',', at) - at, "1e400");
    const fs::path dir = fresh_dir();
    std::ofstream(dir / "in.json") << text;
    std::FILE* out = std::tmpfile();
    std::FILE* err = std::tmpfile();
    CHECK(autan::runtime::run_case(dir / "in.json", dir / "runs",
                                   autan_test::relaxation_registry({}), build, out,
                                   err) == ExitStatus::invalid_input);
    std::fclose(out);
    std::fclose(err);
  }
}

TEST_CASE("A verification case archives its manufactured solution", "[runtime]") {
  // `autan prepare` with verification = Manufactured("euler.mms.json"): every
  // boundary is manufactured, BCType BCTypeUserDefined in solution.cgns.
  const Outcome o = run(input({}, "hybrid-verification.solver-input.json"));
  REQUIRE(o.status == ExitStatus::finished);
  CHECK(read(o.run / "manufactured.json") == read(mms_path));
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  for (const auto& zone : back.face_zones()) {
    if (zone.boundary) CHECK(zone.bc_type == "UserDefined");  // SIDS string of BCTypeUserDefined
  }
  SECTION("E0206: the file differs from its hash") {
    require_error(run(input([](Json& d) { d["verification"]["sha256"] = std::string(64, 'a'); },
                            "hybrid-verification.solver-input.json")),
                  "E0206");
  }
  SECTION("E0207: the file is missing") {
    require_error(
        run(input([](Json& d) { d["verification"]["path"] = "/nonexistent/euler.mms.json"; },
                  "hybrid-verification.solver-input.json")),
        "E0207");
  }
  SECTION("E0705: manufactured boundaries without verification") {
    require_error(run(input([](Json& d) { d["verification"] = nullptr; },
                            "hybrid-verification.solver-input.json")),
                  "E0705");
  }
}

TEST_CASE("log.jsonl records the run's events in the results.md shape", "[runtime]") {
  const Outcome o = run(input(), {.factor = 0.9, .diverge_at = 4});
  std::vector<Json> lines;
  std::ifstream in(o.run / "log.jsonl");
  for (std::string line; std::getline(in, line);) lines.push_back(Json::parse(line));
  REQUIRE(lines.size() >= 4);
  const std::regex time("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}Z");
  const std::regex source("[a-z][a-z0-9-]*");
  for (const Json& l : lines) {
    CHECK(l.size() == 7);
    CHECK(std::regex_match(l["time"].get<std::string>(), time));
    CHECK((l["iteration"].is_null() || l["iteration"].is_number_integer()));
    CHECK((l["level"] == "info" || l["level"] == "warning" || l["level"] == "error"));
    CHECK(std::regex_match(l["source"].get<std::string>(), source));
    CHECK(l["data"].is_object());
  }
  CHECK(lines.front()["iteration"].is_null());
  CHECK(lines[1]["message"] == "relaxation physics ready");  // the physics' own line
  // The run.json diagnostic appears with its code and data.
  const Json info = Json::parse(read(o.run / "run.json"));
  const auto d0107 =
      std::ranges::find_if(lines, [](const Json& l) { return l["code"] == "D0107"; });
  REQUIRE(d0107 != lines.end());
  CHECK((*d0107)["data"] == info["diagnostics"][0]["data"]);
  CHECK((*d0107)["iteration"] == 4);
  CHECK(lines.back()["data"]["status"] == "diverged");
}

TEST_CASE("A continued run goes on bit for bit from the run it continues", "[runtime]") {
  // The state grows at each step, so that the history (forces) and the
  // solution depend on the state handed over; the uninterrupted run is the
  // reference (results.md: same identity parts, same iterations).
  const bool steady = GENERATE(true, false);
  CAPTURE(steady);
  const autan_test::RelaxationOptions options{.growth = 1e-3};
  const auto budget = [&](std::int64_t n) {
    return [steady, n](Json& d) {
      d["reproducibility"] = "run";
      d["monitors"][0]["every"] = 1;
      if (steady)
        d["time"]["max_iterations"] = n;
      else
        d["time"]["steps"] = n;
    };
  };
  const auto doc = [&](std::int64_t n) {
    return steady ? input(budget(n)) : input(budget(n), "hybrid-unsteady.solver-input.json");
  };
  // A chain of three runs. Steady: converges at 28 (see above); measuring
  // from the first iteration of either continuation instead would converge at
  // 38 or later. Unsteady: averaging starts after step 3, in the first run,
  // and goes on through the other two.
  const std::int64_t total = steady ? 10000 : 8;
  const std::int64_t first = steady ? 10 : 4;
  const std::int64_t middle = steady ? 20 : 6;
  const Outcome whole = run(doc(total), options);
  const Outcome a = run(doc(first), options);
  REQUIRE(Json::parse(read(a.run / "run.json"))["status"] == "completed");
  Json next = doc(middle);
  next["start"] = start_from(a.run, "continue");
  const Outcome m = run(next, options);
  REQUIRE(Json::parse(read(m.run / "run.json"))["status"] == "completed");
  Json second = doc(total);
  second["start"] = start_from(m.run, "continue");
  const Outcome b = run(second, options);
  REQUIRE(b.status == ExitStatus::finished);

  const auto all = lines_of(whole.run / "history.jsonl");
  auto chain = lines_of(a.run / "history.jsonl");
  for (const auto& run_dir : {m.run, b.run}) {
    const auto more = lines_of(run_dir / "history.jsonl");
    chain.insert(chain.end(), more.begin(), more.end());
  }
  CHECK(chain == all);
  const Json w = Json::parse(read(whole.run / "run.json"));
  const Json info = Json::parse(read(b.run / "run.json"));
  for (const char* key : {"status", "iterations", "time", "averaging"}) {
    CAPTURE(key);
    CHECK(info[key] == w[key]);
  }
  Json expected_start = second["start"];
  expected_start.erase("run");
  CHECK(info["start"] == expected_start);
  // timing counts this run's iterations only.
  CHECK(info["timing"]["iterations"] ==
        info["iterations"].get<std::int64_t>() - second["start"]["iterations"].get<std::int64_t>());
  // identity: the seven parts, then the start's identity, mode, iterations
  // and solution hash, joined by "\n" (results.md).
  const auto& s = second["start"];
  std::string identity;
  picosha2::hash256_hex_string(second["case"]["sha256"].get<std::string>() + "\n" +
                                   second["mesh"]["sha256"].get<std::string>() + "\n" +
                                   build.commit + "\nfp32+acc64\ncpu\ncpu\n1\n" +
                                   s["identity"].get<std::string>() + "\ncontinue\n" +
                                   std::to_string(s["iterations"].get<int>()) + "\n" +
                                   s["solution_sha256"].get<std::string>(),
                               identity);
  CHECK(info["identity"] == identity);
  for (const char* field : {"Density", "Pressure", "Temperature"}) {
    CAPTURE(field);
    CHECK(solution_field(b.run / "solution.cgns", "FlowSolution", field) ==
          solution_field(whole.run / "solution.cgns", "FlowSolution", field));
    if (!steady) {
      CHECK(solution_field(b.run / "solution.cgns", "FlowSolutionMean", field) ==
            solution_field(whole.run / "solution.cgns", "FlowSolutionMean", field));
    }
  }
}

TEST_CASE("An initial-state file gives each fluid cell its state", "[runtime]") {
  // initial-state.md: rows paired with cells by centroid, whatever their
  // order; the relaxation physics keeps its state (factor 0.5, no growth),
  // so the solution is the file's values cell for cell.
  const Json state = initial_state_document(mesh_path);
  const Json doc = with_initial_state(input(inviscid), state);
  const Outcome o = run(doc);
  REQUIRE(o.status == ExitStatus::finished);
  const auto n = state["cells"].get<std::size_t>();
  const std::array<std::pair<const char*, std::function<double(double)>>, 5> expected{{
      {"Density", [](double c) { return 1.0 + 1e-3 * c; }},
      {"VelocityX", [](double c) { return c; }},
      {"VelocityY", [](double c) { return -c; }},
      {"VelocityZ", [](double) { return 0.5; }},
      {"Pressure", [](double c) { return 1e5 + c; }},
  }};
  for (const auto& [field, value] : expected) {
    CAPTURE(field);
    const auto got = solution_field(o.run / "solution.cgns", "FlowSolution", field);
    REQUIRE(got.size() == n);
    for (std::size_t c = 0; c < n; ++c) CHECK(got[c] == value(static_cast<double>(c)));
  }
  // Archived byte for byte under its hash.
  const std::string sha = doc["initial_state"]["sha256"];
  CHECK(read(o.run / "states" / (sha + ".json")) ==
        read(doc["initial_state"]["path"].get<std::string>()));

  SECTION("E0508: the file differs from its hash") {
    Json bad = doc;
    bad["initial_state"]["sha256"] = std::string(64, '0');
    require_error(run(bad), "E0508");
  }
  SECTION("E0508: not an initial-state document") {
    Json other = state;
    other["variables"] = {"density", "velocity-x", "velocity-y", "velocity-z", "temperature"};
    require_error(run(with_initial_state(input(inviscid), other)), "E0508");
  }
  SECTION("E0508: an unknown key, or a mesh hash that is no SHA-256") {
    Json extra = state;
    extra["note"] = "unknown";
    require_error(run(with_initial_state(input(inviscid), extra)), "E0508");
    Json malformed = state;
    malformed["mesh_sha256"] = "not-a-sha256";
    require_error(run(with_initial_state(input(inviscid), malformed)), "E0508");
  }
  SECTION("E0705: a turbulence model, whose variables the file lacks") {
    require_error(run(with_initial_state(input(), state)), "E0705");
  }
  SECTION("E0509: written for another mesh") {
    Json other = state;
    other["mesh_sha256"] = std::string(64, 'a');
    require_error(run(with_initial_state(input(inviscid), other)), "E0509");
  }
  SECTION("E0510: a row short") {
    Json other = state;
    other["centroids"].erase(other["centroids"].size() - 1);
    other["values"].erase(other["values"].size() - 1);
    other["cells"] = n - 1;
    require_error(run(with_initial_state(input(inviscid), other)), "E0510");
  }
  SECTION("E0511: a row far from every cell, or two rows on one cell") {
    Json far = state;
    far["centroids"][3][0] = far["centroids"][3][0].get<double>() + 1e-3;
    require_error(run(with_initial_state(input(inviscid), far)), "E0511");
    Json twice = state;
    twice["centroids"][3] = twice["centroids"][4];
    require_error(run(with_initial_state(input(inviscid), twice)), "E0511");
  }
  SECTION("E0705: initial_state with start") {
    Json both = doc;
    both["start"] = start_from(o.run, "warm");
    require_error(run(both), "E0705");
  }
}

TEST_CASE("A warm start takes the previous state and counts from 1", "[runtime]") {
  const Outcome a = run(input([](Json& d) { d["time"]["max_iterations"] = 5; }),
                        autan_test::RelaxationOptions{.growth = 0.25});
  Json doc = input([](Json& d) { d["time"]["max_iterations"] = 3; });
  doc["start"] = start_from(a.run, "warm");
  const Outcome b = run(doc);  // growth 0: the state stays the one handed over
  REQUIRE(b.status == ExitStatus::finished);
  const auto lines = history(b.run);
  REQUIRE(lines.size() == 3);
  CHECK(lines[0]["iteration"] == 1);
  const Json info = Json::parse(read(b.run / "run.json"));
  CHECK(info["iterations"] == 3);
  CHECK(info["start"]["mode"] == "warm");
  CHECK(info["start"]["residual_reference"].is_null());
  const auto previous = solution_field(a.run / "solution.cgns", "FlowSolution", "Density");
  CHECK(solution_field(b.run / "solution.cgns", "FlowSolution", "Density") == previous);
  // 1.25^5 times the initial density: not the freestream state.
  CHECK_THAT(previous[0],
             WithinRel(doc["freestream"]["density"].get<double>() * std::pow(1.25, 5.0), 1e-14));
}

TEST_CASE("A start the runtime cannot honour exits 2 before any directory", "[runtime]") {
  const Outcome a = run(input([](Json& d) { d["time"]["max_iterations"] = 4; }));
  const auto from = [&](const std::function<void(Json&)>& edit) {
    Json doc = input();
    doc["start"] = start_from(a.run, "continue");
    edit(doc);
    return run(doc);
  };
  SECTION("the solution changed since the SDK hashed it") {
    require_error(from([](Json& d) { d["start"]["solution_sha256"] = std::string(64, '0'); }),
                  "E0709");
  }
  SECTION("the previous run has no solution") {
    require_error(from([&](Json& d) { d["start"]["run"] = fresh_dir().string(); }), "E0707");
  }
  SECTION("the previous run used another precision") {
    require_error(from([](Json& d) { d["precision"] = "fp64"; }), "E0708");
  }
  SECTION("the previous run solved other variables") {
    require_error(from([](Json& d) {
                    d["start"]["mode"] = "warm";
                    d["start"]["residual_reference"] = nullptr;
                    d["start"]["woken"] = nullptr;
                    d["physics"]["turbulence"]["model"] = "laminar";
                    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
                    for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
                    d["freestream"]["nu_tilde"] = nullptr;
                  }),
                  "E0708");
  }
  SECTION("the previous run was computed on a mesh of other cells") {
    const fs::path hex = data_dir / "mesh" / "hex" / "polyMesh";
    Json doc = input([&](Json& d) {
      d["mesh"]["path"] = fs::absolute(hex).string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(hex);
    });
    doc["start"] = start_from(a.run, "continue");
    require_error(run(doc), "E0203");
  }
  SECTION("the previous solution lacks run metadata the continuation needs") {
    // A solution.cgns whose RunMetadata has no `averaging`, with its hash.
    const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
    Json metadata = Json::parse(read(a.run / "run.json"));
    metadata.erase("averaging");
    autan::io::Solution solution;
    solution.run_metadata = metadata.dump();
    solution.boundary_bc_types.assign(autan::io::meshinfo_zones(mesh).boundary.size(),
                                      "BCFarfield");
    for (const auto& name :
         autan::runtime::field_names(autan::runtime::parse_solver_input(input().dump()))) {
      solution.fields.push_back(
          {.name = name,
           .values = solution_field(a.run / "solution.cgns", "FlowSolution", name.c_str())});
    }
    const fs::path previous = fresh_dir();
    autan::io::write_solution(previous / "solution.cgns", mesh, solution);
    require_error(from([&](Json& d) {
                    d["start"]["run"] = previous.string();
                    d["start"]["solution_sha256"] = file_sha256(previous / "solution.cgns");
                  }),
                  "E0707");
  }
  SECTION("the previous run was computed on another mesh") {
    // The same cells, but another file: its hash differs.
    const fs::path copy = fresh_dir() / "polyMesh";
    fs::copy(mesh_path, copy);
    std::ofstream(copy / "boundary", std::ios::app) << "\n// copy\n";
    Json doc = input([&](Json& d) {
      d["mesh"]["path"] = copy.string();
      d["mesh"]["sha256"] = autan::io::mesh_hash(copy);
    });
    doc["start"] = start_from(a.run, "continue");
    require_error(run(doc), "E0203");
  }
  SECTION("the solver-input rules of start") {
    require_error(from([](Json& d) { d["start"]["residual_reference"] = nullptr; }), "E0705");
    require_error(from([](Json& d) { d["start"]["mode"] = "warm"; }), "E0705");
    require_error(from([](Json& d) { d["time"]["max_iterations"] = 4; }), "E0705");
    require_error(from([](Json& d) { d["start"]["run"] = "relative/run"; }), "E0705");
    require_error(from([](Json& d) { d["start"]["residual_reference"].erase("energy"); }), "E0705");
    require_error(from([](Json& d) { d["start"]["woken"] = nullptr; }), "E0705");
    require_error(from([](Json& d) { d["start"]["woken"] = Json::array({"vorticity"}); }), "E0705");
    require_error(from([](Json& d) { d["start"]["woken"] = Json::array({"energy", "density"}); }),
                  "E0705");
  }
}

TEST_CASE("A periodic mesh runs through its image ghosts", "[runtime][periodic]") {
  // periodic_input(1.0). The relaxation physics
  // checks at every step that the exchange gives each image ghost its
  // owner's value; the run converges as on any mesh (k = 28) and its
  // solution.cgns gives the pairs back.
  const Outcome o = run(periodic_input(1.0));
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);
  // The physics ran on the partition: image ghosts, as many as the
  // relaxation physics reported in its first log line.
  std::ifstream log_in(o.run / "log.jsonl");
  std::int64_t ghosts = -1;
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    if (event["message"] == "relaxation physics ready") ghosts = event["data"]["ghosts"];
  }
  CHECK(ghosts > 0);
  const autan::Mesh back = autan::io::read_mesh(o.run / "solution.cgns");
  CHECK(back.periodic_pairs().size() == 2);
  CHECK(!solution_field(o.run / "solution.cgns", "FlowSolution", "Density").empty());

  require_error(run(periodic_input(2.0)), "E0705");
  // A periodic zone takes no condition and carries no forces monitor.
  require_error(
      run(periodic_input(1.0, [](Json& d) { d["monitors"][0]["zones"] = Json::array({"ymin"}); })),
      "E0705");
  // A periodic zone takes no condition.
  require_error(run(periodic_input(1.0,
                                   [](Json& d) {
                                     Json b = d["boundaries"][0];
                                     b["zone"] = "xmin";
                                     d["boundaries"].push_back(b);
                                   })),
                "E0705");
}

// face-states.md: tests/data/mesh/ringleb, its states written by autan-vv, on the four in-plane
// sides; `edit` changes the document before it is written beside the input.
Json ringleb_input(const std::function<void(Json&)>& edit = {}) {
  const fs::path mesh = data_dir / "mesh" / "ringleb" / "ringleb-n004.msh";
  Json states = Json::parse(read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json"));
  std::string bytes = read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json");
  if (edit) {
    edit(states);
    bytes = states.dump();
  }
  const fs::path file = fresh_dir() / "ringleb.states.json";
  std::ofstream(file, std::ios::binary) << bytes;
  std::string sha;
  picosha2::hash256_hex_string(bytes, sha);
  return input([&](Json& d) {
    d["mesh"]["path"] = fs::absolute(mesh).string();
    d["mesh"]["sha256"] = autan::io::mesh_hash(mesh);
    d["physics"]["turbulence"]["model"] = "inviscid";
    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
    d["freestream"]["nu_tilde"] = nullptr;
    Json zone = d["cell_zones"][0];
    zone["zone"] = "fluid";
    zone["initial"]["nu_tilde"] = nullptr;
    d["cell_zones"] = Json::array({zone});
    Json boundaries = Json::array();
    for (const char* name : {"k-min", "k-max", "inflow", "outflow", "back", "front"}) {
      Json b = d["boundaries"][0];
      for (auto& [key, value] : b.items()) {
        if (key != "zone" && key != "kind") value = nullptr;
      }
      b["zone"] = name;
      const bool sides = std::string(name) != "back" && std::string(name) != "front";
      b["kind"] = sides ? "oracle-state" : "symmetry";
      if (sides) b["states"] = Json{{"path", fs::absolute(file).string()}, {"sha256", sha}};
      boundaries.push_back(b);
    }
    d["boundaries"] = boundaries;
    d["monitors"][0]["zones"] = {"k-min"};
  });
}

TEST_CASE("Face states pair with their zone's faces and are archived with the run", "[runtime]") {
  const Json doc = ringleb_input();
  const Outcome o = run(doc);
  INFO(o.err);
  REQUIRE(o.status == ExitStatus::finished);
  const std::string sha = doc["boundaries"][0]["states"]["sha256"];
  CHECK(read(o.run / "states" / (sha + ".json")) ==
        read(data_dir / "mesh" / "ringleb" / "ringleb-n004.states.json"));
}

TEST_CASE("Face states that do not fit the case exit 2 before any directory", "[runtime]") {
  SECTION("E0420 another file than states.sha256") {
    Json doc = ringleb_input();
    doc["boundaries"][0]["states"]["sha256"] = std::string(64, '0');
    require_error(run(doc), "E0420");
  }
  SECTION("E0420 not a face-state document, whatever zone or key is wrong") {
    require_error(run(ringleb_input([](Json& s) { s["variables"][4] = "temperature"; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["schema"] = nullptr; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["extra"] = 1; })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s.erase("oracle"); })), "E0420");
    require_error(run(ringleb_input([](Json& s) { s["mesh_sha256"] = "abc"; })), "E0420");
    require_error(run(ringleb_input([](Json& s) {
                    s["zones"]["unused"] =
                        Json{{"faces", "two"}, {"centroids", {}}, {"values", {}}};
                  })),
                  "E0420");
  }
  SECTION("E0420 a path that is not a regular file") {
    Json doc = ringleb_input();
    doc["boundaries"][0]["states"]["path"] = fresh_dir().string();
    require_error(run(doc), "E0420");
  }
  SECTION("E0422 exported for another mesh") {
    require_error(run(ringleb_input([](Json& s) { s["mesh_sha256"] = std::string(64, 'a'); })),
                  "E0422");
  }
  SECTION("E0423 a zone missing or with another face count") {
    require_error(run(ringleb_input([](Json& s) { s["zones"].erase("inflow"); })), "E0423");
    require_error(run(ringleb_input([](Json& s) {
                    auto& z = s["zones"]["inflow"];
                    z["faces"] = z["faces"].get<int>() - 1;
                    z["centroids"].erase(z["centroids"].size() - 1);
                    z["values"].erase(z["values"].size() - 1);
                  })),
                  "E0423");
  }
  SECTION("E0421 a centroid away from every face") {
    // The zone spans about 1 m: 1e-6 m is far beyond the 1e-9 x extent tolerance.
    require_error(run(ringleb_input([](Json& s) {
                    auto& c = s["zones"]["k-max"]["centroids"][1];
                    c[0] = c[0].get<double>() + 1e-6;
                  })),
                  "E0421");
  }
}

TEST_CASE("Face states pair on a plane normal to an axis, in any row order", "[runtime]") {
  // The xmin side of the hybrid mesh: every centroid has x = 0. The rows, written here from the
  // mesh's geometry in reverse face order, pair with the faces by centroid (face-states.md).
  const autan::Mesh mesh = autan::io::read_mesh(mesh_path);
  const autan::Geometry g = autan::compute_geometry(mesh);
  autan::index_t zone = -1;
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    if (mesh.face_zones()[z].name == "xmin") zone = static_cast<autan::index_t>(z);
  }
  REQUIRE(zone >= 0);
  const auto off = mesh.boundary_zone_offsets();
  Json centroids = Json::array(), values = Json::array();
  for (auto f = off[static_cast<std::size_t>(zone) + 1];
       f-- > off[static_cast<std::size_t>(zone)];) {
    const auto i = static_cast<std::size_t>(f);
    centroids.push_back({g.face_centroid_x[i], g.face_centroid_y[i], g.face_centroid_z[i]});
    values.push_back({1.2, 100.0, 0.0, 0.0, 1e5});
  }
  const Json states = {
      {"schema", "autan.face-states/1"},
      {"oracle", "uniform"},
      {"mesh_sha256", autan::io::mesh_hash(mesh_path)},
      {"variables", {"density", "velocity-x", "velocity-y", "velocity-z", "pressure"}},
      {"zones",
       {{"xmin", {{"faces", centroids.size()}, {"centroids", centroids}, {"values", values}}}}}};
  const std::string bytes = states.dump();
  const fs::path file = fresh_dir() / "xmin.states.json";
  std::ofstream(file, std::ios::binary) << bytes;
  std::string sha;
  picosha2::hash256_hex_string(bytes, sha);
  const Json doc = input([&](Json& d) {
    d["physics"]["turbulence"]["model"] = "inviscid";
    d["physics"]["turbulence"]["nu_tilde_ratio"] = nullptr;
    d["freestream"]["nu_tilde"] = nullptr;
    for (auto& z : d["cell_zones"]) z["initial"]["nu_tilde"] = nullptr;
    for (auto& b : d["boundaries"]) {
      if (b["kind"] == "wall") b["kind"] = "slip";
      if (b["zone"] == "xmin") {
        b["kind"] = "oracle-state";
        b["states"] = Json{{"path", file.string()}, {"sha256", sha}};
      }
    }
  });
  const Outcome o = run(doc);
  INFO(o.err);
  CHECK(o.status == ExitStatus::finished);
}

TEST_CASE("An equation at round-off from iteration 1 does not hold back convergence", "[runtime]") {
  // Residuals (e + 1) 0.5^(k-1) drop 8 orders at k = 28; momentum-x stays at
  // 1e-15, below the floor 1e-12 x 6 (case-schema.md#time), so it is exempt.
  autan_test::RelaxationOptions quiet;
  quiet.silent_equation = 1;
  const Outcome o = run(input([](Json& d) { d["time"]["max_iterations"] = 100; }), quiet);
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "converged");
  CHECK(info["iterations"] == 28);

  // From iteration 10 it is 1e-3, above the floor: W0710 once, and the run
  // cannot converge on it any more.
  autan_test::RelaxationOptions waking = quiet;
  waking.wake_at = 10;
  const Outcome w = run(input([](Json& d) { d["time"]["max_iterations"] = 100; }), waking);
  REQUIRE(w.status == ExitStatus::finished);
  CHECK(Json::parse(read(w.run / "run.json"))["status"] == "completed");
  const Json info_w = Json::parse(read(w.run / "run.json"));
  REQUIRE(info_w["diagnostics"].size() == 1);
  CHECK(info_w["diagnostics"][0]["code"] == "W0710");
  std::vector<Json> warnings;
  std::ifstream in(w.run / "log.jsonl");
  for (std::string line; std::getline(in, line);) {
    const Json event = Json::parse(line);
    if (event["code"] == "W0710") warnings.push_back(event);
  }
  REQUIRE(warnings.size() == 1);
  CHECK(warnings[0]["level"] == "warning");
  CHECK(warnings[0]["iteration"] == 10);
  CHECK(warnings[0]["data"]["equation"] == "momentum-x");
  CHECK(warnings[0]["data"]["iteration"] == 10);
  CHECK(warnings[0]["data"]["residual"] == 1e-3);
  CHECK(warnings[0]["data"]["floor"] == 1e-12 * 6.0);
}

TEST_CASE("A continuation keeps the equations that left their floor", "[runtime]") {
  // momentum-x leaves its floor at iterations 10-14, then falls back: it is
  // no longer exempt, so the run cannot converge (budget 40). A run
  // continued from iteration 20 must do the same, through start.woken.
  autan_test::RelaxationOptions options;
  options.silent_equation = 1;
  options.wake_at = 10;
  options.sleep_at = 15;
  const auto doc = [](std::int64_t n) {
    return input([n](Json& d) {
      d["reproducibility"] = "run";
      d["time"]["max_iterations"] = n;
    });
  };
  const Outcome whole = run(doc(40), options);
  const Outcome a = run(doc(20), options);
  Json next = doc(40);
  next["start"] = start_from(a.run, "continue");
  CHECK(next["start"]["woken"] == Json::array({"momentum-x"}));
  const Outcome b = run(next, options);
  REQUIRE(b.status == ExitStatus::finished);
  const Json info = Json::parse(read(b.run / "run.json"));
  CHECK(info["status"] == "completed");
  CHECK(info["start"]["woken"] == Json::array({"momentum-x"}));
  // The equation woke in the first run: no new W0710 in the continuation.
  CHECK(info["diagnostics"].empty());
  auto chain = lines_of(a.run / "history.jsonl");
  const auto rest = lines_of(b.run / "history.jsonl");
  chain.insert(chain.end(), rest.begin(), rest.end());
  CHECK(chain == lines_of(whole.run / "history.jsonl"));
}

TEST_CASE("In fp32+acc64 an equation converges at its floor at the latest", "[runtime]") {
  // Residuals (e + 1) 0.5^(k-1); scales 1e4 (e + 1) give floors
  // 0.5 2^-23 1e4 (e + 1) = 5.96e-4 (e + 1), reached at k = 12 for every
  // equation, before the 8-order drop at k = 28 (case-schema.md#time).
  const std::string precision = GENERATE("fp32+acc64", "fp64");
  CAPTURE(precision);
  autan_test::RelaxationOptions scaled;
  for (int e = 0; e < 6; ++e) scaled.scales.push_back(1e4 * (e + 1));
  const Outcome o = run(input([&](Json& d) { d["precision"] = precision; }), scaled);
  REQUIRE(o.status == ExitStatus::finished);
  const Json info = Json::parse(read(o.run / "run.json"));
  CHECK(info["status"] == "converged");
  if (precision == "fp64") {
    CHECK(info["iterations"] == 28);
    CHECK(info["convergence"].is_null());
    return;
  }
  CHECK(info["iterations"] == 12);
  const std::vector<std::string> equations{"density",    "momentum-x", "momentum-y",
                                           "momentum-z", "energy",     "nu-tilde"};
  REQUIRE(info["convergence"]["floor"].size() == equations.size());
  for (std::size_t e = 0; e < equations.size(); ++e) {
    CHECK(info["convergence"]["floor"][equations[e]] ==
          std::ldexp(0.5, -23) * 1e4 * static_cast<double>(e + 1));
  }

  // A physics without scales has no floor; one with the wrong count fails.
  const Outcome plain = run(input());
  CHECK(Json::parse(read(plain.run / "run.json"))["convergence"].is_null());
  autan_test::RelaxationOptions short_scales;
  short_scales.scales = {1.0};
  const Outcome bad = run(input(), short_scales);
  CHECK(bad.status == ExitStatus::failed);
  const Json failed = Json::parse(read(bad.run / "run.json"));
  CHECK(failed["status"] == "failed");
  CHECK(failed["diagnostics"][0]["code"] == "E0704");
}

#ifdef AUTAN_MPIEXEC
namespace {

// The arrays of solution.cgns's WallSolution subregion, by name.
std::map<std::string, std::vector<double>> wall_solution(const fs::path& file) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  int regions = 0;
  REQUIRE(cg_nsubregs(fn, 1, 1, &regions) == CG_OK);
  std::map<std::string, std::vector<double>> out;
  for (int r = 1; r <= regions; ++r) {
    char name[33];
    int dim = 0, bc_len = 0, gc_len = 0;
    CGNS_ENUMT(GridLocation_t) location{};
    CGNS_ENUMT(PointSetType_t) ptset{};
    cgsize_t npnts = 0;
    REQUIRE(cg_subreg_info(fn, 1, 1, r, name, &dim, &location, &ptset, &npnts, &bc_len, &gc_len) ==
            CG_OK);
    if (std::string(name) != "WallSolution") continue;
    REQUIRE(cg_goto(fn, 1, "Zone_t", 1, "ZoneSubRegion_t", r, "end") == CG_OK);
    int arrays = 0;
    REQUIRE(cg_narrays(&arrays) == CG_OK);
    for (int a = 1; a <= arrays; ++a) {
      char array[33];
      CGNS_ENUMT(DataType_t) type{};
      int rank = 0;
      cgsize_t size[3];
      REQUIRE(cg_array_info(a, array, &type, &rank, size) == CG_OK);
      if (std::string(array) == "PointList") continue;
      std::vector<double> values(static_cast<std::size_t>(size[0]));
      REQUIRE(cg_array_read_as(a, CGNS_ENUMV(RealDouble), values.data()) == CG_OK);
      out[array] = std::move(values);
    }
  }
  cg_close(fn);
  return out;
}

// The relaxation physics' colour sweeps redone on the whole mesh, from the
// global colouring: each step, `sweeps` symmetric sweeps over the colours,
// each cell of a colour taking the mean of itself and its face neighbours,
// summed in increasing global id; the field starts at global id mod 7. With
// `groups` (RelaxationOptions::groups), a line smoother: a group is one node
// of the colouring, greedy by increasing least global id, and its cells are
// swept in decreasing id; this colouring is computed here, not by the code
// under test.
std::vector<double> swept(const autan::Mesh& mesh, int steps, int sweeps, int groups = 0) {
  const auto ids = mesh.cell_global_id();
  const auto z = [](auto i) { return static_cast<std::size_t>(i); };
  const std::int64_t period = 2 * std::int64_t{groups};
  const auto node = [&](autan::index_t c) -> std::int64_t {
    return groups > 0 && ids[z(c)] % period < groups ? ids[z(c)] % period
                                                     : std::int64_t{groups} + ids[z(c)];
  };
  std::map<std::int64_t, std::vector<autan::index_t>> members;  // node -> cells, by id
  std::map<std::int64_t, std::set<std::int64_t>> around_node;
  for (autan::index_t c = 0; c < mesh.n_cells(); ++c) members[node(c)].push_back(c);
  for (autan::index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    const auto a = node(mesh.owner()[z(f)]), b = node(mesh.neighbour()[z(f)]);
    if (a != b) {
      around_node[a].insert(b);
      around_node[b].insert(a);
    }
  }
  std::vector<std::pair<std::int64_t, std::int64_t>> by_least;  // (least id, node)
  for (const auto& [n, cells] : members) {
    std::int64_t least = ids[z(cells.front())];
    for (const auto c : cells) least = std::min(least, ids[z(c)]);
    by_least.emplace_back(least, n);
  }
  std::ranges::sort(by_least);
  std::map<std::int64_t, std::int32_t> node_colour;
  for (const auto& [least, n] : by_least) {
    std::set<std::int32_t> taken;
    for (const auto m : around_node[n]) {
      if (node_colour.contains(m)) taken.insert(node_colour[m]);
    }
    std::int32_t k = 0;
    while (taken.contains(k)) ++k;
    node_colour[n] = k;
  }
  std::int32_t n_colours = 0;
  for (const auto& [n, k] : node_colour) n_colours = std::max(n_colours, k + 1);
  std::vector<double> w(z(mesh.n_cells()));
  for (std::size_t c = 0; c < w.size(); ++c) w[c] = static_cast<double>(ids[c] % 7);
  const auto colour_sweep = [&](std::int32_t k) {
    std::vector<autan::index_t> cells;
    for (const auto& [least, n] : by_least) {
      if (node_colour[n] != k || n >= groups) continue;
      auto line = members[n];
      std::ranges::sort(line, std::greater{}, [&](autan::index_t c) { return ids[z(c)]; });
      cells.insert(cells.end(), line.begin(), line.end());
    }
    for (autan::index_t c = 0; c < mesh.n_cells(); ++c) {
      if (node(c) >= groups && node_colour[node(c)] == k) cells.push_back(c);
    }
    for (const autan::index_t c : cells) {
      std::vector<std::pair<std::int64_t, double>> around{
          {ids[static_cast<std::size_t>(c)], w[static_cast<std::size_t>(c)]}};
      for (const autan::index_t f : mesh.cell_faces(c)) {
        if (f >= mesh.n_interior_faces()) continue;
        const auto i = static_cast<std::size_t>(f);
        const autan::index_t o = mesh.owner()[i] == c ? mesh.neighbour()[i] : mesh.owner()[i];
        around.emplace_back(ids[static_cast<std::size_t>(o)], w[static_cast<std::size_t>(o)]);
      }
      std::ranges::sort(around);
      double sum = 0.0;
      for (const auto& [id, value] : around) sum += value;
      w[static_cast<std::size_t>(c)] = sum / static_cast<double>(around.size());
    }
  };
  for (int step = 0; step < steps; ++step) {
    for (int s = 0; s < sweeps; ++s) {
      for (std::int32_t k = 0; k < n_colours; ++k) colour_sweep(k);
      for (std::int32_t k = n_colours - 1; k >= 0; --k) colour_sweep(k);
    }
  }
  return w;
}

// autan-run-relaxation under the launcher on `ranks` ranks: its run
// directory (none for an invalid input), after checking its exit status;
// `diagnostic`, when given, receives the JSON diagnostic line of stderr.
fs::path run_on_ranks(const Json& doc, int ranks, const std::vector<std::string>& options,
                      int expected_status = 0, Json* diagnostic = nullptr,
                      const std::vector<std::string>& extra_args = {}, bool directory = true) {
  const fs::path dir = fresh_dir();
  const fs::path input_file = dir / "solver-input.json";
  // One device per rank (solver-input.md, `devices`), unless a test sets it.
  Json with_count = doc;
  if (with_count["devices"]["count"] == 1) with_count["devices"]["count"] = ranks;
  std::ofstream(input_file) << with_count.dump();
  std::vector<std::string> args{AUTAN_MPIEXEC,
                                AUTAN_MPIEXEC_NUMPROC_FLAG,
                                std::to_string(ranks),
                                AUTAN_RUN_RELAXATION,
                                "--input",
                                input_file.string(),
                                "--runs",
                                (dir / "runs").string()};
  args.insert(args.end(), extra_args.begin(), extra_args.end());
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  std::vector<std::string> env = options;
  for (char** e = environ; *e != nullptr; ++e) env.emplace_back(*e);
  std::vector<char*> envp;
  for (auto& e : env) envp.push_back(e.data());
  envp.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  const std::string err_file = (dir / "stderr.txt").string();
  posix_spawn_file_actions_addopen(&actions, 2, err_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                                   0644);
  pid_t pid = 0;
  REQUIRE(posix_spawn(&pid, AUTAN_MPIEXEC, &actions, nullptr, argv.data(), envp.data()) == 0);
  posix_spawn_file_actions_destroy(&actions);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == expected_status);
  if (diagnostic != nullptr) {
    std::ifstream in(err_file);
    for (std::string line; std::getline(in, line);) {
      if (line.starts_with("{")) *diagnostic = Json::parse(line);
    }
  }
  fs::path run;
  if (fs::exists(dir / "runs")) {
    for (const auto& entry : fs::directory_iterator(dir / "runs")) run = entry.path();
  }
  CHECK(run.empty() == (expected_status == 2 || !directory));
  return run;
}

}  // namespace

TEST_CASE("Several ranks give the bits of one", "[runtime][mpi][gpu]") {
  // Fields and residuals vary from cell to cell, so that a cell gathered to
  // the wrong place or a norm summed in another order would change bits;
  // forces are recorded at every iteration, and the state grows.
  const bool periodic = GENERATE(false, true);
  const int ranks = GENERATE(2, 4);
  CAPTURE(periodic, ranks);
  const auto edit = [](Json& d) {
    d["reproducibility"] = "run";
    d["monitors"][0]["every"] = 1;
    d["time"]["max_iterations"] = 12;
  };
  const Json doc = periodic ? periodic_input(1.0, edit) : input(edit);
  const autan_test::RelaxationOptions options{
      .factor = 0.9, .pressure_offset = 1000.0, .growth = 1e-3, .vary = true};
  const Outcome one = run(doc, options);
  REQUIRE(one.status == ExitStatus::finished);
  const fs::path many = run_on_ranks(doc, ranks,
                                     {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                                      "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});

  CHECK(lines_of(many / "history.jsonl") == lines_of(one.run / "history.jsonl"));
  // Reproducibility "ranks": a partition even on one rank, and face keys that
  // match the mesh file on 1 and N ranks (the physics checks them); a run on
  // several ranks in another mode says its results depend on the count.
  Json keyed = doc;
  keyed["reproducibility"] = "ranks";
  autan_test::RelaxationOptions checking = options;
  checking.check_keys = true;
  const Outcome one_keyed = run(keyed, checking);
  REQUIRE(one_keyed.status == ExitStatus::finished);
  const fs::path many_keyed =
      run_on_ranks(keyed, ranks,
                   {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                    "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_CHECK_KEYS=1"});
  CHECK(Json::parse(read(many_keyed / "run.json"))["status"] == "completed");
  CHECK(lines_of(many_keyed / "history.jsonl") == lines_of(one_keyed.run / "history.jsonl"));
  CHECK(solution_field(many_keyed / "solution.cgns", "FlowSolution", "Pressure") ==
        solution_field(one_keyed.run / "solution.cgns", "FlowSolution", "Pressure"));
  const auto has_event = [](const fs::path& run_dir) {
    std::ifstream in(run_dir / "log.jsonl");
    for (std::string line; std::getline(in, line);) {
      if (Json::parse(line)["data"].contains("mode")) return true;
    }
    return false;
  };
  CHECK(has_event(many));
  CHECK_FALSE(has_event(many_keyed));
  // A smoother sweeping the global colours (PhysicsContext::cell_colours),
  // the ghosts exchanged after each colour, and an argmin over the ranks:
  // the history of one rank.
  autan_test::RelaxationOptions sweeping = options;
  sweeping.colour_sweeps = 2;
  const Outcome one_swept = run(doc, sweeping);
  const fs::path many_swept =
      run_on_ranks(doc, ranks,
                   {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                    "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_COLOUR_SWEEPS=2"});
  CHECK(lines_of(one_swept.run / "history.jsonl") != lines_of(one.run / "history.jsonl"));
  CHECK(lines_of(many_swept / "history.jsonl") == lines_of(one_swept.run / "history.jsonl"));
  // The swept field, cell for cell: on one rank, as an independent sweep of
  // the whole mesh gives it (12 steps of 2 symmetric sweeps); on several,
  // the same bits.
  const auto swept_one =
      solution_field(one_swept.run / "solution.cgns", "FlowSolution", "TurbulentSANuTilde");
  CHECK(solution_field(many_swept / "solution.cgns", "FlowSolution", "TurbulentSANuTilde") ==
        swept_one);
  if (!periodic) CHECK(swept_one == swept(autan::io::read_mesh(mesh_path), 12, 2));
  // A line smoother: the physics' groups (each spanning the mesh) stay whole
  // on one rank, and it sweeps the colours of its groups, each in its order;
  // on several ranks, the bits of one, and on one, those of an independent
  // sweep of the whole mesh.
  autan_test::RelaxationOptions lining = sweeping;
  lining.groups = ranks;
  const Outcome one_lined = run(doc, lining);
  REQUIRE(one_lined.status == ExitStatus::finished);
  const fs::path many_lined =
      run_on_ranks(doc, ranks,
                   {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                    "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1", "AUTAN_TEST_COLOUR_SWEEPS=2",
                    "AUTAN_TEST_GROUPS=" + std::to_string(ranks)});
  CHECK(lines_of(many_lined / "history.jsonl") == lines_of(one_lined.run / "history.jsonl"));
  const auto lined_one =
      solution_field(one_lined.run / "solution.cgns", "FlowSolution", "TurbulentSANuTilde");
  CHECK(lined_one != swept_one);
  CHECK(solution_field(many_lined / "solution.cgns", "FlowSolution", "TurbulentSANuTilde") ==
        lined_one);
  if (!periodic) CHECK(lined_one == swept(autan::io::read_mesh(mesh_path), 12, 2, ranks));
  // Groups the partition refuses end every rank before any directory, with their code.
  Json malformed;
  CHECK(run_on_ranks(doc, ranks, {"AUTAN_TEST_GROUPS=-1"}, 1, &malformed, {}, false).empty());
  CHECK(malformed["code"] == "E0140");
  // Rank 1 alone fails while the others go on into the partition: every rank ends.
  Json alone;
  CHECK(run_on_ranks(doc, ranks, {"AUTAN_TEST_GROUPS=-2"}, 1, &alone, {}, false).empty());
  CHECK(alone["code"] == "E0140");
  CHECK(alone["data"]["rank"] == 1);
#ifdef AUTAN_HAVE_CUDA
  // On CUDA devices, the ranks sharing the host's GPUs: the relaxation
  // physics checks at every step that the exchange fills its ghosts in
  // device memory, through a CUDA-aware MPI; the history stays that of one rank.
  Json on_gpu = doc;
  on_gpu["devices"]["kind"] = "cuda";
  const fs::path gpu = run_on_ranks(on_gpu, ranks,
                                    {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                                     "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});
  CHECK(lines_of(gpu / "history.jsonl") == lines_of(one.run / "history.jsonl"));
#endif
  const Json a = Json::parse(read(one.run / "run.json"));
  const Json b = Json::parse(read(many / "run.json"));
  for (const char* key : {"status", "iterations", "time", "diagnostics"}) {
    CAPTURE(key);
    CHECK(a[key] == b[key]);
  }
  for (const char* field : {"Density", "VelocityX", "Pressure", "TurbulentSANuTilde"}) {
    CAPTURE(field);
    const auto expected = solution_field(one.run / "solution.cgns", "FlowSolution", field);
    CHECK(!expected.empty());
    CHECK(solution_field(many / "solution.cgns", "FlowSolution", field) == expected);
  }
  // An initial-state file, split among the ranks by their owned cells: the
  // solution of one rank, cell for cell.
  if (!periodic) {
    Json plain = doc;
    inviscid(plain);
    const Json from_file = with_initial_state(plain, initial_state_document(mesh_path));
    const Outcome one_file = run(from_file, options);
    const fs::path many_file =
        run_on_ranks(from_file, ranks,
                     {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_PRESSURE_OFFSET=1000",
                      "AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"});
    for (const char* field : {"Density", "VelocityY", "Pressure"}) {
      CAPTURE(field);
      CHECK(solution_field(many_file / "solution.cgns", "FlowSolution", field) ==
            solution_field(one_file.run / "solution.cgns", "FlowSolution", field));
    }
  }
  const auto wall = wall_solution(one.run / "solution.cgns");
  CHECK(wall.contains("Pressure"));
  CHECK(wall_solution(many / "solution.cgns") == wall);
  // Rank 0 logged the partition: every cell owned once.
  std::int64_t owned = 0;
  std::ifstream log_in(many / "log.jsonl");
  for (std::string line; std::getline(log_in, line);) {
    const Json event = Json::parse(line);
    if (event["source"] == "mpi") {
      for (const auto& n : event["data"]["owned_cells"]) owned += n.get<std::int64_t>();
    }
  }
  CHECK(owned == autan::io::read_mesh(doc["mesh"]["path"].get<std::string>()).n_cells());

  // Every cell from 7 on diverges, on several ranks: the least is reported,
  // as a single rank names it.
  const Outcome diverged = run(doc, {.factor = 0.9, .diverge_at = 5, .diverge_cell = 7});
  const Json d = Json::parse(read(run_on_ranks(doc, ranks,
                                               {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_DIVERGE_AT=5",
                                                "AUTAN_TEST_DIVERGE_CELL=7"}) /
                                  "run.json"));
  CHECK(d["status"] == "diverged");
  CHECK(d["diagnostics"] == Json::parse(read(diverged.run / "run.json"))["diagnostics"]);

  // A physics registered for one rank refuses several before any directory,
  // and a scope refuses by the rank count it is given.
  CHECK(run_on_ranks(doc, ranks, {"AUTAN_TEST_ONE_RANK=1"}, 2).empty());
  // Every rank's scope must see the count for the run to start.
  CHECK(!run_on_ranks(doc, ranks, {"AUTAN_TEST_EXPECT_RANKS=" + std::to_string(ranks)}).empty());
  Json refusal;
  CHECK(run_on_ranks(doc, ranks, {"AUTAN_TEST_EXPECT_RANKS=" + std::to_string(ranks + 1)}, 2,
                     &refusal)
            .empty());
  CHECK(refusal["code"] == "E0706");
  CHECK(refusal["data"] == Json{{"ranks", ranks}, {"path", "physics"}});

  // A physics failure on the last rank alone, the others waiting for it in a
  // collective call, ends every rank with status 1 and a failed run.json.
  const Json f =
      Json::parse(read(run_on_ranks(doc, ranks,
                                    {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_THROW_AT=4",
                                     "AUTAN_TEST_THROW_RANK=" + std::to_string(ranks - 1)},
                                    1) /
                       "run.json"));
  CHECK(f["status"] == "failed");
  CHECK(f["iterations"] == 3);
  CHECK(f["diagnostics"][0]["code"] == "E0704");
  CHECK(f["diagnostics"][0]["data"]["rank"] == ranks - 1);

  // The same failure in the final monitor record (loads() called only by
  // finish, `every` beyond the run), the others inside the collective sum of
  // their loads(): every rank ends, none waits forever (ctest TIMEOUT).
  Json final_only = doc;
  final_only["monitors"][0]["every"] = 1000;
  const Json g =
      Json::parse(read(run_on_ranks(final_only, ranks,
                                    {"AUTAN_TEST_FACTOR=0.5", "AUTAN_TEST_THROW_IN_LOADS=1",
                                     "AUTAN_TEST_THROW_RANK=" + std::to_string(ranks - 1)},
                                    1) /
                       "run.json"));
  CHECK(g["status"] == "failed");
  CHECK(g["monitors"].is_null());
  CHECK(g["diagnostics"][0]["code"] == "E0704");
  CHECK(g["diagnostics"][0]["data"]["rank"] == ranks - 1);

  // A stop file the last rank creates during iteration 5: rank 0 sees it, and
  // all stop before iteration 6.
  const fs::path stop_file = fresh_dir() / "stop";
  const fs::path by_file = run_on_ranks(doc, ranks,
                                        {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_TOUCH_AT=5",
                                         "AUTAN_TEST_TOUCH_RANK=" + std::to_string(ranks - 1),
                                         "AUTAN_TEST_TOUCH_FILE=" + stop_file.string()},
                                        130, nullptr, {"--stop-file", stop_file.string()});
  CHECK(Json::parse(read(by_file / "run.json"))["status"] == "stopped");
  CHECK(Json::parse(read(by_file / "run.json"))["iterations"] == 5);

  // A signal reaching the last rank alone, after iteration 5, stops all of
  // them before iteration 6, with the solution written.
  const fs::path stopped = run_on_ranks(doc, ranks,
                                        {"AUTAN_TEST_FACTOR=0.9", "AUTAN_TEST_SIGNAL_AT=5",
                                         "AUTAN_TEST_SIGNAL_RANK=" + std::to_string(ranks - 1)},
                                        130);
  const Json st = Json::parse(read(stopped / "run.json"));
  CHECK(st["status"] == "stopped");
  CHECK(st["iterations"] == 5);
  CHECK(fs::exists(stopped / "solution.cgns"));
}

TEST_CASE("Several ranks average and continue as one", "[runtime][mpi]") {
  // Unsteady with averaging after step 3: two runs of 4 then 4 steps on
  // several ranks, the second continuing the first, give the 8 steps of one
  // rank: the start state is split among the ranks and the mean and restart
  // fields gathered back.
  const int ranks = GENERATE(2, 4);
  const std::string mode = GENERATE("run", "ranks");
  CAPTURE(ranks, mode);
  const auto steps = [mode](std::int64_t n) {
    return [n, mode](Json& d) {
      d["reproducibility"] = mode;
      d["reproducibility"] = "run";
      d["monitors"][0]["every"] = 1;
      d["time"]["steps"] = n;
    };
  };
  const std::vector<std::string> env{"AUTAN_TEST_GROWTH=0.001", "AUTAN_TEST_VARY=1"};
  const Outcome whole =
      run(input(steps(8), "hybrid-unsteady.solver-input.json"), {.growth = 1e-3, .vary = true});
  const fs::path a = run_on_ranks(input(steps(4), "hybrid-unsteady.solver-input.json"), ranks, env);
  Json next = input(steps(8), "hybrid-unsteady.solver-input.json");
  next["start"] = start_from(a, "continue");
  const fs::path b = run_on_ranks(next, ranks, env);

  auto chain = lines_of(a / "history.jsonl");
  const auto more = lines_of(b / "history.jsonl");
  chain.insert(chain.end(), more.begin(), more.end());
  CHECK(chain == lines_of(whole.run / "history.jsonl"));
  for (const char* solution : {"FlowSolution", "FlowSolutionMean"}) {
    for (const char* field : {"Density", "Pressure"}) {
      CAPTURE(solution, field);
      const auto expected = solution_field(whole.run / "solution.cgns", solution, field);
      CHECK(!expected.empty());
      CHECK(solution_field(b / "solution.cgns", solution, field) == expected);
    }
  }
}
#endif
