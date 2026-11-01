#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"
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
  std::int64_t touch_at = 0;  ///< iteration after which `touch_rank` creates `touch_file`; 0: never
  int touch_rank = 0;
  std::string touch_file{};
  std::int64_t log_bytes = 0;  ///< a log event of this many bytes at every step; 0: none
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
  /// Symmetric sweeps per step of a smoother in the global colours (PhysicsContext::cell_colours):
  /// each cell of a colour takes the mean of itself and its face neighbours, summed in increasing
  /// global id, then the ghosts are exchanged; the residuals scale by 1 + 1e-3 of the field's RMS.
  int colour_sweeps = 0;
  /// Bytes of host memory the physics allocates, touches and keeps from its
  /// construction on.
  std::int64_t hold_bytes = 0;
  int expect_ranks = 0;  ///< its scope refuses a run on another number of ranks; 0: any
  /// An equation the flow does not excite: its residual stays at round-off
  /// (1e-15), except from iteration `wake_at` (0: never) to `sleep_at`
  /// (excluded; 0: never), where it is 1e-3.
  std::int64_t silent_equation = -1;
  std::int64_t wake_at = 0;
  std::int64_t sleep_at = 0;     ///< 0: never; from then on, back at 1e-15
  std::vector<double> scales{};  ///< residual_scales(); empty: none
  /// Groups of cells the partition keeps whole: cell c in group c mod (2 groups) when below
  /// `groups`, so each group spans the mesh, its cells in decreasing order. The physics throws
  /// unless the runtime asked for them, every group it holds a cell of is wholly owned, and
  /// PhysicsContext gives it exactly those, in that order, each of one colour. With
  /// colour_sweeps, the sweeps run over the group colours, each group in its order: a line
  /// smoother. The runtime must ask once per rank. 0: none; -1: a malformed group, cell 0
  /// twice; -2: rank 1 alone (by the launcher's rank variable) throws E0140 when asked.
  int groups = 0;

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
    read("AUTAN_TEST_COLOUR_SWEEPS", o.colour_sweeps);
    read("AUTAN_TEST_HOLD_BYTES", o.hold_bytes);
    read("AUTAN_TEST_EXPECT_RANKS", o.expect_ranks);
    read("AUTAN_TEST_THROW_RANK", o.throw_rank);
    read("AUTAN_TEST_THROW_IN_LOADS", o.throw_in_loads);
    read("AUTAN_TEST_SIGNAL_AT", o.signal_at);
    read("AUTAN_TEST_SIGNAL_RANK", o.signal_rank);
    read("AUTAN_TEST_GROUPS", o.groups);
    read("AUTAN_TEST_TOUCH_AT", o.touch_at);
    read("AUTAN_TEST_TOUCH_RANK", o.touch_rank);
    if (const char* v = std::getenv("AUTAN_TEST_TOUCH_FILE")) o.touch_file = v;
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
    held_.assign(static_cast<std::size_t>(options.hold_bytes), 1);
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
    if (iteration == options_.touch_at && options_.touch_rank == context_.ranks.rank())
      std::ofstream(options_.touch_file).put('\n');
    if (options_.log_bytes > 0) {
      context_.log.write(
          "info", "solver", "bulk", "",
          "{\"blob\": \"" + std::string(static_cast<std::size_t>(options_.log_bytes), 'x') + "\"}");
    }
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
    // Cells diverge on every rank that owns one; each rank reports its
    // least, owned cells being in mesh order.
    if (iteration == options_.diverge_at) {
      const auto& mesh = context_.mesh;
      for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c) {
        if (mesh.cell_global_id()[static_cast<std::size_t>(c)] < options_.diverge_cell) continue;
        state_[0].values[static_cast<std::size_t>(c)] = std::numeric_limits<double>::quiet_NaN();
        if (!report.diverged_cell) report.diverged_cell = c;
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
    const bool lines = options_.groups > 0;
    const auto& colours = lines ? context_.cell_group_colours : context_.cell_colours;
    const std::int32_t n_colours = lines ? context_.n_group_colours : context_.n_colours;
    const auto& offsets = context_.group_offsets;
    std::vector<char> in_group(static_cast<std::size_t>(mesh.n_owned_cells()), 0);
    for (const autan::index_t c : context_.group_cells) in_group[static_cast<std::size_t>(c)] = 1;
    const auto colour_sweep = [&](std::int32_t k) {
      // Each group of colour k in its order, then the other cells of colour k.
      std::vector<autan::index_t> cells;
      for (std::size_t g = 0; lines && g + 1 < offsets.size(); ++g) {
        if (colours[static_cast<std::size_t>(
                context_.group_cells[static_cast<std::size_t>(offsets[g])])] != k)
          continue;
        for (auto i = offsets[g]; i < offsets[g + 1]; ++i)
          cells.push_back(context_.group_cells[static_cast<std::size_t>(i)]);
      }
      for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c) {
        if (in_group[static_cast<std::size_t>(c)] == 0 && colours[static_cast<std::size_t>(c)] == k)
          cells.push_back(c);
      }
      for (const autan::index_t c : cells) {
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
      for (std::int32_t k = 0; k < n_colours; ++k) colour_sweep(k);
      for (std::int32_t k = n_colours - 1; k >= 0; --k) colour_sweep(k);
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
  std::vector<char> held_;  // hold_bytes
};

// The groups the runtime gives the physics (RelaxationOptions::groups): each
// group it holds a cell of wholly owned and given, in decreasing global id,
// groups by increasing least id, one colour per group.
inline void check_groups(const autan::runtime::PhysicsContext& context, int n,
                         const std::vector<std::int64_t>& sizes) {
  const auto fail = [](const std::string& why) { throw std::runtime_error("groups: " + why); };
  if (sizes.empty()) fail("the runtime did not ask for them");
  const auto& mesh = context.mesh;
  const auto ids = mesh.cell_global_id();
  const auto at = [](auto i) { return static_cast<std::size_t>(i); };
  std::int64_t grouped = 0;
  for (autan::index_t c = 0; c < mesh.n_owned_cells(); ++c)
    grouped += ids[at(c)] % (2 * n) < n ? 1 : 0;
  const auto& offsets = context.group_offsets;
  if (offsets.empty() || offsets.back() != grouped || at(grouped) != context.group_cells.size())
    fail("not every owned grouped cell is given once");
  std::int64_t last_least = -1;
  for (std::size_t g = 0; g + 1 < offsets.size(); ++g) {
    const auto first = context.group_cells[at(offsets[g])];
    const auto k = ids[at(first)] % (2 * n);
    if (offsets[g + 1] - offsets[g] != sizes[at(k)]) fail("a group is split or merged");
    for (auto i = offsets[g]; i < offsets[g + 1]; ++i) {
      const auto c = context.group_cells[at(i)];
      if (c < 0 || c >= mesh.n_owned_cells() || ids[at(c)] % (2 * n) != k) fail("a wrong cell");
      if (i > offsets[g] && ids[at(c)] >= ids[at(context.group_cells[at(i - 1)])])
        fail("not in the physics' order");
      if (context.cell_group_colours[at(c)] != context.cell_group_colours[at(first)])
        fail("two colours in one group");
    }
    const auto least = ids[at(context.group_cells[at(offsets[g + 1] - 1)])];
    if (least <= last_least) fail("groups not by increasing least id");
    last_least = least;
  }
}

inline autan::runtime::PhysicsRegistry relaxation_registry(RelaxationOptions options) {
  autan::runtime::PhysicsRegistry registry;
  // Cells per group, recorded when the runtime asks for the groups, and how many times it did.
  auto group_sizes = std::make_shared<std::vector<std::int64_t>>();
  auto asked = std::make_shared<int>(0);
  autan::runtime::PartitionGroups groups;
  if (options.groups == -1) {
    groups = [](const autan::runtime::SolverInput&, const autan::Mesh&, const autan::Geometry&) {
      return autan::runtime::CellGroups{.offsets = {0, 2}, .cells = {0, 0}};
    };
  }
  if (options.groups == -2) {
    groups = [](const autan::runtime::SolverInput&, const autan::Mesh&, const autan::Geometry&) {
      const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
      if (rank == nullptr) rank = std::getenv("PMI_RANK");
      if (rank == nullptr) throw std::runtime_error("no launcher rank");
      if (std::string(rank) == "1")
        throw autan::Error(autan::ErrorCode::partition_invalid_request, "rank 1 refuses");
      return autan::runtime::CellGroups{};
    };
  }
  if (options.groups > 0) {
    groups = [n = options.groups, group_sizes, asked](const autan::runtime::SolverInput&,
                                                      const autan::Mesh& mesh,
                                                      const autan::Geometry&) {
      ++*asked;
      autan::runtime::CellGroups g;
      g.offsets.push_back(0);
      group_sizes->assign(static_cast<std::size_t>(n), 0);
      for (int k = 0; k < n; ++k) {
        for (autan::index_t c = k; c < mesh.n_cells(); c += 2 * n) {
          g.cells.insert(g.cells.begin() + g.offsets.back(), c);
          ++(*group_sizes)[static_cast<std::size_t>(k)];
        }
        g.offsets.push_back(static_cast<autan::offset_t>(g.cells.size()));
      }
      return g;
    };
  }
  registry.add(
      {.kind = "*", .turbulence = "*", .precision = "*", .device = "*"},
      [options, group_sizes, asked](const autan::runtime::PhysicsContext& context) {
        if (options.groups > 0) {
          if (*asked != 1) throw std::runtime_error("groups asked for " + std::to_string(*asked));
          check_groups(context, options.groups, *group_sizes);
        }
        return std::make_unique<Relaxation>(context, options);
      },
      [options](const autan::runtime::SolverInput& /*input*/,
                int ranks) -> std::optional<autan::runtime::PhysicsRefusal> {
        if (options.expect_ranks == 0 || ranks == options.expect_ranks) return std::nullopt;
        return autan::runtime::PhysicsRefusal{
            .message = "refused on " + std::to_string(ranks) + " ranks",
            .path = "physics",
            .data_json = "{\"ranks\": " + std::to_string(ranks) + "}"};
      },
      !options.one_rank, groups);
  return registry;
}

}  // namespace autan_test
