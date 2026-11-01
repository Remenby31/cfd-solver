#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "autan/io/read_mesh.hpp"
#include "autan/runtime/physics.hpp"

#ifdef AUTAN_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace autan_test {

// Behaviour of the relaxation physics, so that tests can drive every path of
// the runtime: residual decay factor per iteration, an iteration at which a
// cell turns non-finite or the physics throws, and a pause per iteration.
struct RelaxationOptions {
  double factor = 0.5;
  std::int64_t diverge_at = 0;      ///< 0: never
  autan::index_t diverge_cell = 0;  ///< every cell from this mesh index on diverges
  std::int64_t throw_at = 0;        ///< 0: never
  int throw_rank = -1;              ///< the rank that throws; -1: every rank
  std::int64_t signal_at = 0;  ///< iteration after which `signal_rank` raises SIGTERM; 0: never
  int signal_rank = 0;
  int pause_ms = 0;
  double pressure_offset = 0.0;  ///< added to every fluid cell's pressure, Pa
  double growth = 0.0;           ///< every field is multiplied by 1 + growth at each step
  bool signal_on_start = false;  ///< raise SIGTERM once constructed, before iteration 1
  bool throw_in_loads = false;
  std::int64_t nan_loads_from = 0;  ///< loads are NaN from this iteration on; 0: never
  bool throw_in_fields = false;
  /// Every field and residual varies from cell to cell: values scaled by
  /// 1 + 1e-6 g with g the cell's mesh index, residuals by their RMS.
  bool vary = false;
  bool one_rank = false;  ///< registered as running on one rank only
  /// Checks PhysicsContext's face keys against the mesh file: each local face's global face has
  /// the local face's cells (by global id), swapped when it is marked reversed.
  bool check_keys = false;
  std::string refuse_code{};  ///< its scope refuses every input with this code; empty: none
  int expect_ranks = 0;       ///< its scope refuses a run on another number of ranks; 0: any
  /// Symmetric sweeps per step of a smoother in the global colours (PhysicsContext::cell_colours):
  /// each cell of a colour takes the mean of itself and its face neighbours, summed in increasing
  /// global id, then the ghosts are exchanged; the residuals scale by 1 + 1e-3 of the field's RMS.
  int colour_sweeps = 0;
  /// An equation the flow does not excite: its residual stays at round-off
  /// (1e-15), except from iteration `wake_at` (0: never) to `sleep_at`
  /// (excluded; 0: never), where it is 1e-3.
  std::int64_t silent_equation = -1;
  std::int64_t wake_at = 0;
  std::int64_t sleep_at = 0;     ///< 0: never; from then on, back at 1e-15
  std::vector<double> scales{};  ///< residual_scales(); empty: none

  static RelaxationOptions from_environment() {
    RelaxationOptions o;
    auto read = [](const char* name, auto& out) {
      if (const char* v = std::getenv(name))
        out = static_cast<std::remove_reference_t<decltype(out)>>(std::atof(v));
    };
    read("AUTAN_TEST_FACTOR", o.factor);
    read("AUTAN_TEST_DIVERGE_AT", o.diverge_at);
    read("AUTAN_TEST_THROW_AT", o.throw_at);
    read("AUTAN_TEST_PAUSE_MS", o.pause_ms);
    read("AUTAN_TEST_DIVERGE_CELL", o.diverge_cell);
    read("AUTAN_TEST_PRESSURE_OFFSET", o.pressure_offset);
    read("AUTAN_TEST_GROWTH", o.growth);
    read("AUTAN_TEST_VARY", o.vary);
    read("AUTAN_TEST_ONE_RANK", o.one_rank);
    read("AUTAN_TEST_CHECK_KEYS", o.check_keys);
    read("AUTAN_TEST_COLOUR_SWEEPS", o.colour_sweeps);
    read("AUTAN_TEST_EXPECT_RANKS", o.expect_ranks);
    read("AUTAN_TEST_THROW_RANK", o.throw_rank);
    read("AUTAN_TEST_THROW_IN_LOADS", o.throw_in_loads);
    read("AUTAN_TEST_SIGNAL_AT", o.signal_at);
    read("AUTAN_TEST_SIGNAL_RANK", o.signal_rank);
    return o;
  }
};

// An implementation of the runtime's Physics interface that solves nothing:
// the state starts at each zone's `initial` (or the start's fields), grows by
// `growth` per step, and every residual decays as r_k = factor^(k-1). It exists to exercise the
// runtime, the run directory and solution.cgns through the same seam the real physics use.
class Relaxation final : public autan::runtime::Physics {
 public:
  Relaxation(const autan::runtime::PhysicsContext& context, RelaxationOptions options)
      : context_(context), options_(options) {
    const auto& mesh = context.mesh;
    std::map<std::string, const autan::runtime::CellZoneInput*> zones;
    for (const auto& z : context.input.cell_zones) zones[z.zone] = &z;
    const auto names = autan::runtime::field_names(context.input);
    for (const auto& name : names) state_.push_back({.name = name, .values = {}});
    for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c) {
      const auto& zone = *zones.at(
          mesh.cell_zones()[static_cast<std::size_t>(mesh.cell_zone()[static_cast<std::size_t>(c)])]
              .name);
      const auto& s = zone.initial;
      const bool fluid = s.kind == "fluid";
      for (auto& f : state_) {
        double v = 0.0;
        if (f.name == "Density") v = fluid ? *s.density : *zone.medium.density;
        if (f.name == "VelocityX" && fluid) v = (*s.velocity)[0];
        if (f.name == "VelocityY" && fluid) v = (*s.velocity)[1];
        if (f.name == "VelocityZ" && fluid) v = (*s.velocity)[2];
        if (f.name == "Pressure" && fluid) v = *s.pressure + options.pressure_offset;
        if (f.name == "Temperature") v = s.temperature.value_or(0.0);
        if (f.name == "TurbulentSANuTilde" && fluid) v = s.nu_tilde.value_or(0.0);
        if (f.name == "TurbulentEnergyKinetic" && fluid) v = s.k.value_or(0.0);
        if (f.name == "TurbulentDissipationRate" && fluid) v = s.omega.value_or(0.0);
        f.values.push_back(v);
      }
    }
    // An initial-state file sets the fluid cells' density, velocity and pressure.
    if (context.initial != nullptr) {
      for (const auto& from : *context.initial) {
        for (auto& f : state_) {
          if (f.name != from.name) continue;
          for (std::size_t c = 0; c < f.values.size(); ++c) {
            if (!std::isnan(from.values[c])) f.values[c] = from.values[c];
          }
        }
      }
    }
    if (options_.vary) {
      for (auto& f : state_) {
        for (std::size_t c = 0; c < f.values.size(); ++c) f.values[c] *= scale(c);
      }
      // RMS of the scales over the mesh: a sum over every rank's cells.
      std::array<autan::runtime::ExactSum, 2> sums;
      for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_owned_cells()); ++c) {
        sums[0].add(scale(c) * scale(c));
        sums[1].add(1.0);
      }
      context_.ranks.sum(sums);
      rms_ = std::sqrt(sums[0].value() / sums[1].value());
    }
    if (options_.check_keys) check_keys();
    mean_ = state_;
    context_.log.write("info", "solver", "relaxation physics ready", "",
                       "{\"ghosts\": " + std::to_string(mesh.n_cells() - mesh.n_owned_cells()) +
                           ", \"device\": " + std::to_string(context.device) +
                           ", \"stream\": " + (context.stream != nullptr ? "true" : "false") + "}");
    for (auto& f : mean_) std::fill(f.values.begin(), f.values.end(), 0.0);
    if (context.start != nullptr) {
      state_ = context.start->fields;
      // RestartState: the averaging sums, then the weight (one value per cell).
      const auto& restart = context.start->restart_fields;
      if (!restart.empty()) {
        for (std::size_t f = 0; f < mean_.size(); ++f) mean_[f].values = restart[f].values;
        weight_ = restart.back().values.at(0);
      }
    }
    if (options_.signal_on_start) std::raise(SIGTERM);
  }

  autan::runtime::StepReport step(std::int64_t iteration) override {
    last_stepped_iteration_ = iteration;
    if (options_.pause_ms > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(options_.pause_ms));
    if (iteration == options_.throw_at &&
        (options_.throw_rank < 0 || options_.throw_rank == context_.ranks.rank()))
      throw std::runtime_error("relaxation: requested failure");
    if (iteration == options_.signal_at && options_.signal_rank == context_.ranks.rank())
      std::raise(SIGTERM);
    autan::runtime::StepReport report;
    // Equation e decays from e + 1, so that equations are told apart.
    const double r = std::pow(options_.factor, static_cast<double>(iteration - 1));
    const auto equations = autan::runtime::residual_names(context_.input).size();
    const double smoothed = options_.colour_sweeps > 0 ? 1.0 + 1e-3 * sweep() : 1.0;
    for (std::size_t e = 0; e < equations; ++e)
      report.residuals.push_back(static_cast<double>(e + 1) * r * rms_ * smoothed);
    if (options_.silent_equation >= 0) {
      const bool awake = options_.wake_at > 0 && iteration >= options_.wake_at &&
                         (options_.sleep_at == 0 || iteration < options_.sleep_at);
      report.residuals[static_cast<std::size_t>(options_.silent_equation)] = awake ? 1e-3 : 1e-15;
    }
    for (auto& f : state_) {
      for (auto& v : f.values) v *= 1.0 + options_.growth;
    }
    check_ghosts();
    const auto& time = context_.input.time;
    if (time.kind == "unsteady") report.time_step = time.dt.value_or(1e-3);
    // Cells diverge on every rank that owns one; each rank reports the one of
    // least mesh index, whatever the order of its cells.
    if (iteration == options_.diverge_at) {
      const auto& mesh = context_.mesh;
      const auto id = [&](autan::index_t c) {
        return mesh.cell_global_id()[static_cast<std::size_t>(c)];
      };
      for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c) {
        if (id(c) < options_.diverge_cell) continue;
        state_[0].values[static_cast<std::size_t>(c)] = std::numeric_limits<double>::quiet_NaN();
        if (!report.diverged_cell || id(c) < id(*report.diverged_cell)) report.diverged_cell = c;
      }
    }
    return report;
  }

  // The exchange must give every ghost (a periodic image) its owner's value:
  // a field of global ids, ghosts unset, then compared after the exchange.
  void check_ghosts() const {
    const auto& mesh = context_.mesh;
    const auto n = static_cast<std::size_t>(mesh.n_cells());
    std::vector<double> id(n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_owned_cells()); ++c)
      id[c] = static_cast<double>(mesh.cell_global_id()[c]);
    double* const component = id.data();
    context_.ghosts.forward(std::span<double* const>(&component, 1), autan::MemorySpace::host,
                            nullptr);
    for (std::size_t c = 0; c < n; ++c) {
      if (id[c] != static_cast<double>(mesh.cell_global_id()[c]))
        throw std::runtime_error("relaxation: a ghost did not receive its owner's value");
    }
#ifdef AUTAN_HAVE_CUDA
    // On a CUDA run, the same through device memory on the run's stream.
    if (context_.stream == nullptr) return;
    for (std::size_t c = static_cast<std::size_t>(mesh.n_owned_cells()); c < n; ++c)
      id[c] = std::numeric_limits<double>::quiet_NaN();
    const auto stream = static_cast<cudaStream_t>(context_.stream);
    double* device = nullptr;
    if (cudaMalloc(&device, n * sizeof(double)) != cudaSuccess)
      throw std::runtime_error("relaxation: cudaMalloc");
    cudaMemcpyAsync(device, id.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream);
    context_.ghosts.forward(std::span<double* const>(&device, 1), autan::MemorySpace::cuda,
                            context_.stream);
    cudaMemcpyAsync(id.data(), device, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    const bool done = cudaStreamSynchronize(stream) == cudaSuccess;
    cudaFree(device);
    if (!done) throw std::runtime_error("relaxation: the device exchange failed");
    for (std::size_t c = 0; c < n; ++c) {
      if (id[c] != static_cast<double>(mesh.cell_global_id()[c]))
        throw std::runtime_error("relaxation: a device ghost did not receive its owner's value");
    }
#endif
  }

  void accumulate_average(double weight) override {
    weight_ += weight;
    for (std::size_t f = 0; f < state_.size(); ++f) {
      for (std::size_t c = 0; c < state_[f].values.size(); ++c) {
        mean_[f].values[c] += weight * state_[f].values[c];
      }
    }
  }

  [[nodiscard]] std::vector<double> residual_scales() const override { return options_.scales; }

  [[nodiscard]] std::vector<autan::io::Field> fields(bool mean) const override {
    if (options_.throw_in_fields) throw std::runtime_error("relaxation: requested fields failure");
    if (!mean && !smooth_.empty()) {
      // The swept field, for the tests to compare cell by cell.
      auto out = state_;
      for (auto& f : out) {
        if (f.name == "TurbulentSANuTilde")
          f.values.assign(smooth_.begin(), smooth_.begin() + context_.mesh.n_owned_cells());
      }
      return out;
    }
    if (!mean) return state_;
    auto out = mean_;
    for (auto& f : out) {
      for (auto& v : f.values) v /= weight_;
    }
    return out;
  }

  [[nodiscard]] std::vector<autan::io::Field> restart_fields() const override {
    auto out = mean_;
    for (auto& f : out) f.name = "Sum" + f.name;
    out.push_back({.name = "AverageWeight",
                   .values = std::vector<double>(state_.front().values.size(), weight_)});
    return out;
  }

  [[nodiscard]] std::vector<autan::io::Field> wall_fields(
      std::span<const autan::index_t> faces) const override {
    std::vector<autan::io::Field> out{{.name = "Pressure", .values = {}},
                                      {.name = "WallShearStressX", .values = {}},
                                      {.name = "WallShearStressY", .values = {}},
                                      {.name = "WallShearStressZ", .values = {}}};
    for (const auto f : faces) {
      out[0].values.push_back(pressure(context_.mesh.owner()[static_cast<std::size_t>(f)]));
      for (std::size_t k = 1; k < 4; ++k) out[k].values.push_back(0.0);
    }
    return out;
  }

  [[nodiscard]] autan::runtime::Loads loads(std::span<const autan::index_t> zones,
                                            const std::array<double, 3>& center) const override {
    // Pressure force (p - p_inf) S over the zones' faces, its moment about
    // `center`; the cell's pressure stands for the face's.
    if (options_.throw_in_loads &&
        (options_.throw_rank < 0 || options_.throw_rank == context_.ranks.rank()))
      throw std::runtime_error("relaxation: requested loads failure");
    // Exact sums over this rank's faces, then over the ranks: the same bits
    // on any number of ranks.
    std::array<autan::runtime::ExactSum, 6> sums;
    const auto& mesh = context_.mesh;
    const auto& g = context_.geometry;
    const double p_inf = context_.input.freestream.pressure;
    for (const auto z : zones) {
      const auto offsets = mesh.boundary_zone_offsets();
      for (auto f = offsets[static_cast<std::size_t>(z)];
           f < offsets[static_cast<std::size_t>(z) + 1]; ++f) {
        const auto i = static_cast<std::size_t>(f);
        const double dp = pressure(mesh.owner()[i]) - p_inf;
        const std::array<double, 3> force{dp * g.face_area_x[i], dp * g.face_area_y[i],
                                          dp * g.face_area_z[i]};
        const std::array<double, 3> arm{g.face_centroid_x[i] - center[0],
                                        g.face_centroid_y[i] - center[1],
                                        g.face_centroid_z[i] - center[2]};
        for (std::size_t k = 0; k < 3; ++k) sums[k].add(force[k]);
        sums[3].add(arm[1] * force[2] - arm[2] * force[1]);
        sums[4].add(arm[2] * force[0] - arm[0] * force[2]);
        sums[5].add(arm[0] * force[1] - arm[1] * force[0]);
      }
    }
    context_.ranks.sum(sums);
    autan::runtime::Loads loads;
    for (std::size_t k = 0; k < 3; ++k) {
      loads.force[k] = sums[k].value();
      loads.moment[k] = sums[3 + k].value();
    }
    if (options_.nan_loads_from > 0 && last_stepped_iteration_ >= options_.nan_loads_from)
      loads.force[1] = std::numeric_limits<double>::quiet_NaN();
    return loads;
  }

 private:
  [[nodiscard]] double pressure(autan::index_t cell) const {
    for (const auto& f : state_) {
      if (f.name == "Pressure") return f.values[static_cast<std::size_t>(cell)];
    }
    return 0.0;
  }

  // colour_sweeps symmetric sweeps over the colours, the ghosts exchanged
  // after each colour; the RMS of the field over every rank's cells, exactly.
  double sweep() {
    const auto& mesh = context_.mesh;
    const auto n = static_cast<std::size_t>(mesh.n_cells());
    if (smooth_.empty()) {
      smooth_.assign(n, 0.0);
      for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_owned_cells()); ++c)
        smooth_[c] = static_cast<double>(mesh.cell_global_id()[c] % 7);
      exchange();
    }
    const auto ids = mesh.cell_global_id();
    const auto colour_sweep = [&](std::int32_t k) {
      for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c) {
        if (context_.cell_colours[static_cast<std::size_t>(c)] != k) continue;
        std::vector<std::pair<std::int64_t, double>> around{
            {ids[static_cast<std::size_t>(c)], smooth_[static_cast<std::size_t>(c)]}};
        for (const autan::index_t f : mesh.cell_faces(c)) {
          if (f >= mesh.n_interior_faces()) continue;
          const auto i = static_cast<std::size_t>(f);
          const autan::index_t o = mesh.owner()[i] == c ? mesh.neighbour()[i] : mesh.owner()[i];
          around.emplace_back(ids[static_cast<std::size_t>(o)],
                              smooth_[static_cast<std::size_t>(o)]);
        }
        std::ranges::sort(around);
        double sum = 0.0;
        for (const auto& [id, value] : around) sum += value;
        smooth_[static_cast<std::size_t>(c)] = sum / static_cast<double>(around.size());
      }
      exchange();
    };
    for (int s = 0; s < options_.colour_sweeps; ++s) {
      for (std::int32_t k = 0; k < context_.n_colours; ++k) colour_sweep(k);
      for (std::int32_t k = context_.n_colours - 1; k >= 0; --k) colour_sweep(k);
    }
    std::array<autan::runtime::ExactSum, 2> sums;
    for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_owned_cells()); ++c) {
      sums[0].add(smooth_[c] * smooth_[c]);
      sums[1].add(1.0);
    }
    context_.ranks.sum(sums);
    // The least value over the mesh and the least id holding it (values are
    // repeated, so ties decide): its id enters the result too.
    double least = std::numeric_limits<double>::infinity();
    std::int64_t at_least = std::numeric_limits<std::int64_t>::max();
    for (std::size_t c = 0; c < static_cast<std::size_t>(mesh.n_owned_cells()); ++c) {
      if (smooth_[c] < least || (smooth_[c] == least && ids[c] < at_least)) {
        least = smooth_[c];
        at_least = ids[c];
      }
    }
    const auto [value, id] = context_.ranks.argmin(least, at_least);
    return std::sqrt(sums[0].value() / sums[1].value()) + 1e-6 * value +
           1e-9 * static_cast<double>(id);
  }

  void exchange() {
    double* const component = smooth_.data();
    context_.ghosts.forward(std::span<double* const>(&component, 1), autan::MemorySpace::host,
                            nullptr);
  }

  void check_keys() const {
    const autan::Mesh global = autan::io::read_mesh(context_.input.mesh_path);
    const auto& mesh = context_.mesh;
    const auto ids = mesh.cell_global_id();
    if (context_.face_global_id.size() != static_cast<std::size_t>(mesh.n_faces()) ||
        context_.face_reversed.size() != context_.face_global_id.size())
      throw std::runtime_error("relaxation: face keys of another size than the mesh's faces");
    // The geometry is the mesh file's, bit for bit: each owned cell's by its
    // global id, each face between owned cells (or of one, on the boundary)
    // by its key, the area vector negated if reversed. Faces of a ghost may
    // sit on a periodic image, translated.
    const autan::Geometry file = autan::compute_geometry(global);
    const auto& geometry = context_.geometry;
    for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c) {
      const auto i = static_cast<std::size_t>(c);
      const auto k = static_cast<std::size_t>(ids[i]);
      if (geometry.cell_volume[i] != file.cell_volume[k] ||
          geometry.cell_centroid_x[i] != file.cell_centroid_x[k])
        throw std::runtime_error("relaxation: a cell's geometry is not the mesh file's");
    }
    for (autan::index_t f = 0; f < mesh.n_faces(); ++f) {
      const auto i = static_cast<std::size_t>(f);
      const auto g = static_cast<std::size_t>(context_.face_global_id[i]);
      const bool reversed = context_.face_reversed[i] != 0;
      const bool owned =
          mesh.owner()[i] < mesh.n_owned_cells() &&
          (f >= mesh.n_interior_faces() || mesh.neighbour()[i] < mesh.n_owned_cells());
      const double sign = reversed ? -1.0 : 1.0;
      if (owned && (geometry.face_area[i] != file.face_area[g] ||
                    geometry.face_area_x[i] != sign * file.face_area_x[g] ||
                    geometry.face_area_z[i] != sign * file.face_area_z[g] ||
                    geometry.face_centroid_y[i] != file.face_centroid_y[g]))
        throw std::runtime_error("relaxation: a face's geometry is not the mesh file's");
      const auto owner = ids[static_cast<std::size_t>(mesh.owner()[i])];
      if (f >= mesh.n_interior_faces()) {
        if (reversed ||
            owner != global.cell_global_id()[static_cast<std::size_t>(global.owner()[g])])
          throw std::runtime_error("relaxation: a boundary face's key does not match");
        continue;
      }
      // A face to a periodic image has a pair's face of zone a for key: its
      // neighbour is the owner of the matched face of zone b.
      const auto neighbour = ids[static_cast<std::size_t>(mesh.neighbour()[i])];
      const auto global_owner =
          global.cell_global_id()[static_cast<std::size_t>(global.owner()[g])];
      std::int64_t global_neighbour = -1;
      if (g < static_cast<std::size_t>(global.n_interior_faces())) {
        global_neighbour = global.cell_global_id()[static_cast<std::size_t>(global.neighbour()[g])];
      } else {
        const auto offsets = global.boundary_zone_offsets();
        for (std::size_t p = 0; p < global.periodic_pairs().size(); ++p) {
          const auto first = static_cast<std::size_t>(
              offsets[static_cast<std::size_t>(global.periodic_pairs()[p].zone_a)]);
          const auto partners = global.periodic_partners(p);
          if (g >= first && g < first + partners.size()) {
            global_neighbour = global.cell_global_id()[static_cast<std::size_t>(
                global.owner()[static_cast<std::size_t>(partners[g - first])])];
          }
        }
        if (global_neighbour < 0)
          throw std::runtime_error("relaxation: a periodic face's key is no zone-a face");
      }
      if ((reversed ? neighbour : owner) != global_owner ||
          (reversed ? owner : neighbour) != global_neighbour)
        throw std::runtime_error("relaxation: an interior face's key does not match");
    }
  }

  [[nodiscard]] double scale(std::size_t cell) const {
    return 1.0 + 1e-6 * static_cast<double>(context_.mesh.cell_global_id()[cell]);
  }

  autan::runtime::PhysicsContext context_;
  RelaxationOptions options_;
  double rms_ = 1.0;
  std::vector<double> smooth_;  // the field colour_sweeps smooths
  std::int64_t last_stepped_iteration_ = 0;
  std::vector<autan::io::Field> state_;
  std::vector<autan::io::Field> mean_;
  double weight_ = 0.0;
};

inline autan::runtime::PhysicsRegistry relaxation_registry(RelaxationOptions options) {
  autan::runtime::PhysicsRegistry registry;
  registry.add(
      {.kind = "*", .turbulence = "*", .precision = "*", .device = "*"},
      [options](const autan::runtime::PhysicsContext& context) {
        return std::make_unique<Relaxation>(context, options);
      },
      [options](const autan::runtime::SolverInput& /*input*/,
                int ranks) -> std::optional<autan::runtime::PhysicsRefusal> {
        if (options.refuse_code.empty() == false) {
          return autan::runtime::PhysicsRefusal{.message = "refused with its own code",
                                                .path = "physics",
                                                .data_json = "{}",
                                                .code = options.refuse_code};
        }
        if (options.expect_ranks == 0 || ranks == options.expect_ranks) return std::nullopt;
        return autan::runtime::PhysicsRefusal{
            .message = "refused on " + std::to_string(ranks) + " ranks",
            .path = "physics",
            .data_json = "{\"ranks\": " + std::to_string(ranks) + "}"};
      },
      !options.one_rank);
  return registry;
}

}  // namespace autan_test
