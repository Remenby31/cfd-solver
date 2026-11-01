#pragma once

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/cgns.hpp"
#include "autan/partition/partition.hpp"
#include "autan/runtime/input.hpp"
#include "autan/runtime/ranks.hpp"

namespace autan::runtime {

/// What one iteration (steady) or one time step (unsteady) reports.
struct StepReport {
  /// One norm per equation of Physics::equations(), defined as in
  /// docs/contracts/results.md#history (FP64, owned fluid cells).
  std::vector<double> residuals;
  /// Physical time advanced by an unsteady step, in seconds; 0 when steady.
  double time_step = 0.0;
  /// First cell whose state or residual is non-finite or non-physical; set
  /// when the solution diverged.
  std::optional<index_t> diverged_cell;
};

/// Force and moment the fluid exerts on a monitor's surfaces (results.md,
/// forces monitor): F (N) and M about the monitor's moment centre (N·m), in
/// mesh axes. The runtime derives the coefficients.
struct Loads {
  std::array<double, 3> force{};
  std::array<double, 3> moment{};
};

/// The run's log (log.jsonl of results.md): one event per call, stamped
/// with the wall-clock time and the current iteration by the runtime.
class Log {
 public:
  virtual ~Log() = default;
  /// `level`: info, warning or error; `source`: [a-z][a-z0-9-]* (solver,
  /// linear, io, mpi…); `code`: a registry code or empty; `data_json`: a
  /// JSON object.
  virtual void write(std::string_view level, std::string_view source, std::string_view message,
                     std::string_view code = {}, std::string_view data_json = "{}") = 0;
};

/// The state of the previous run a run starts from (input.start), read from
/// its solution.cgns.
struct StartState {
  std::vector<io::Field> fields;  ///< FlowSolution: field_names(input), in order
  /// `continue` only (empty for `warm`): FlowSolutionMean when that run
  /// averaged, and its RestartState.
  std::vector<io::Field> mean_fields;
  std::vector<io::Field> restart_fields;
};

/// Fills the ghost cells of PhysicsContext::mesh from their owners: the
/// cells of other ranks' parts and periodic images. Each component holds n_cells
/// values (owned then ghost) in `space`, at most 16 per call; `stream` is a
/// cudaStream_t or null. Call it before any loop that reads ghost values.
class GhostExchange {
 public:
  virtual ~GhostExchange() = default;
  virtual void forward(std::span<float* const> components, MemorySpace space, void* stream) = 0;
  virtual void forward(std::span<double* const> components, MemorySpace space, void* stream) = 0;
};

/// Valid until the run ends; a physics may keep a copy.
struct PhysicsContext {
  const Mesh& mesh;
  const Geometry& geometry;
  const SolverInput& input;
  Log& log;
  const StartState* start;  ///< null unless input.start
  GhostExchange& ghosts;
  /// The ranks of the run; norms and loads reduce over them.
  Ranks& ranks;
  /// Per face zone (index into mesh.face_zones()) with an oracle-state condition, the states of
  /// its owned boundary faces in mesh order, component-major: density, velocity x, y, z,
  /// pressure (SI), 5 x the zone's faces (face-states.md, paired by the runtime).
  const std::map<index_t, std::vector<double>>& boundary_states;
  /// input.initial_state: Density, VelocityX/Y/Z and Pressure (SI) of the owned cells, paired
  /// from the initial-state file (initial-state.md); NaN in solid cells, which keep their zone's
  /// state. Null without one; the physics then starts from the zones' `initial`.
  const std::vector<io::Field>* initial = nullptr;
  /// devices.kind = cuda: this rank's CUDA device, current on the calling thread, and its
  /// stream (a cudaStream_t) for the physics' work and ghost exchanges. -1 and null on the CPU.
  int device = -1;
  void* stream = nullptr;
  /// For each face of `mesh`, its face in the mesh the run read and whether it runs against it
  /// (1: owner and neighbour swapped, area vector negated): with mesh.cell_global_id() and
  /// mesh.ghost_translation(), the keys a physics orders its sums by under reproducibility
  /// "ranks", so that its bits do not depend on the partition (solver-input.md).
  std::span<const std::int64_t> face_global_id;
  std::span<const std::uint8_t> face_reversed;
  /// The colour of each cell of `mesh` (owned and ghosts) in the global distance-1 colouring of
  /// the mesh (colour_cells: neighbours across faces and periodic pairs never share one), and
  /// the number of colours: the same on any number of ranks and in any cell order, so a
  /// smoother that sweeps them, exchanging ghosts after each colour, repeats its bits.
  std::span<const std::int32_t> cell_colours;
  std::int32_t n_colours = 0;
  /// With a physics that gives partition groups (PhysicsRegistry::Entry::groups): the groups
  /// this rank owns, as CSR over the owned cells of `mesh`, by increasing least
  /// cell_global_id, each group's cells in the order the physics gave them; and the colour of
  /// each cell of `mesh` (owned and ghosts) in the global colouring of the groups
  /// (colour_cells with groups: a cell outside every group is its own node), with the number of
  /// colours. A line smoother that sweeps these colours, each line in its order, exchanging
  /// ghosts after each colour, repeats its bits on any number of ranks. Empty otherwise.
  std::span<const offset_t> group_offsets;
  std::span<const index_t> group_cells;
  std::span<const std::int32_t> cell_group_colours;
  std::int32_t n_group_colours = 0;
};

// case-schema.md#time: C * eps32 with C = 0.5, at least 4.3 times the round-off plateaus
// measured on a NACA 0012 slice, manufactured solutions and Ni's transonic bump.
inline constexpr double round_off_floor = 0.5 * 0x1p-23;

/// A discretisation of the case's equations. The runtime owns the iteration
/// loop, convergence, signals and outputs; an implementation owns its state
/// arrays (host or device), its kernels and their schedule. Constructed from
/// a PhysicsContext that outlives it; the state starts at the cell zones'
/// `initial` values, or at `context.start` when the run starts from another
/// (docs/contracts/runtime.md).
class Physics {
 public:
  virtual ~Physics() = default;
  /// Advances one pseudo-time iteration (steady) or one time step (unsteady).
  virtual StepReport step(std::int64_t iteration) = 0;
  /// Adds the current state to the time average with weight `weight` (the
  /// step's time step), for unsteady runs with average_from.
  virtual void accumulate_average(double weight) = 0;
  /// Cell-centred fields with the SIDS names of results.md (field_names()),
  /// current state or time average.
  [[nodiscard]] virtual std::vector<io::Field> fields(bool mean) const = 0;
  /// A constant flux scale per equation of residual_names(input), in that
  /// order, finite and >= 0; empty for none. Called once, after
  /// construction: in fp32+acc64, an equation converges at a residual of
  /// round_off_floor times its scale at the latest (docs/contracts/runtime.md).
  [[nodiscard]] virtual std::vector<double> residual_scales() const = 0;
  /// Cell-centred state beyond fields(false) that a `continue` needs to go
  /// on bit for bit (earlier time levels, averaging sums); empty when none.
  [[nodiscard]] virtual std::vector<io::Field> restart_fields() const = 0;
  /// Pressure and WallShearStressX/Y/Z on `wall_faces` (ascending).
  [[nodiscard]] virtual std::vector<io::Field> wall_fields(
      std::span<const index_t> wall_faces) const = 0;
  /// Loads on the boundary faces of `zones` (indices into Mesh::face_zones()).
  [[nodiscard]] virtual Loads loads(std::span<const index_t> zones,
                                    const std::array<double, 3>& moment_center) const = 0;
};

/// What a physics implementation can run: each member matches the input's
/// value, or "*" any value.
struct PhysicsKey {
  std::string kind;        ///< physics.kind
  std::string turbulence;  ///< physics.turbulence.model
  std::string precision;   ///< precision
  std::string device;      ///< devices.kind
};

using PhysicsFactory = std::function<std::unique_ptr<Physics>(const PhysicsContext&)>;

/// Why a physics whose key matches an input cannot run it: the runtime reports it as E0706,
/// before any directory exists.
struct PhysicsRefusal {
  std::string message;
  std::string path;              ///< document path of the offending key; also `data.path`
  std::string data_json = "{}";  ///< machine-readable details: the text of a JSON object
  /// The registry code reported: E0706 (no implementation for the input) unless another code
  /// says more, as E1109 for a reproducibility the physics cannot honour.
  std::string code = "E0706";
};

/// What a physics supports beyond its key (media, zones, conditions, the number of ranks the
/// run spans): nullopt when it can run the input on `ranks` ranks. A scope must not throw: it
/// reports what it refuses as a PhysicsRefusal.
using PhysicsScope =
    std::function<std::optional<PhysicsRefusal>(const SolverInput& input, int ranks)>;

/// Groups of cells of the global mesh that the partition keeps on one rank, as CSR
/// (PartitionOptions::group_offsets and group_cells).
struct CellGroups {
  std::vector<offset_t> offsets;
  std::vector<index_t> cells;
};

/// The groups a physics needs whole on one rank (a smoother's lines), from the global mesh and
/// its geometry. Every rank calls it once with the same mesh before partitioning, on any number
/// of ranks, so it must be deterministic; invalid groups are E0140. The physics receives them
/// as PhysicsContext::group_offsets and group_cells, coloured.
using PartitionGroups =
    std::function<CellGroups(const SolverInput& input, const Mesh& mesh, const Geometry& geometry)>;

/// The physics a build of autan-run provides; the first matching key wins.
class PhysicsRegistry {
 public:
  struct Entry {
    PhysicsFactory factory;
    PhysicsScope scope;  ///< empty: every input the key matches
    /// Whether it runs on several ranks: its norms, loads and other sums
    /// reduce over PhysicsContext::ranks (runtime.md, Ranks).
    bool several_ranks = false;
    PartitionGroups groups;  ///< empty: no group
  };
  void add(PhysicsKey key, PhysicsFactory factory, PhysicsScope scope = {},
           bool several_ranks = false, PartitionGroups groups = {});
  /// The entry for `input`, or null when this build has none (E0706).
  [[nodiscard]] const Entry* find(const SolverInput& input) const;

 private:
  std::vector<std::pair<PhysicsKey, Entry>> entries_;
};

}  // namespace autan::runtime
