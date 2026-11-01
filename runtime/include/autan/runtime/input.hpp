#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace autan::runtime {

/// The solver input of docs/contracts/solver-input.md (autan.solver-input/1),
/// every value in SI. Optional members are the keys the contract allows to be
/// null.
struct Viscosity {
  std::string law;  ///< "sutherland" or "constant"
  std::optional<double> mu_ref, t_ref, s, mu;
};

struct Medium {
  std::string kind;  ///< "ideal-gas", "constant-density-fluid" or "solid"
  std::string name;
  std::optional<double> gas_constant, gamma, prandtl;  // ideal gas
  std::optional<Viscosity> viscosity;                  // fluids
  std::optional<double> density;                       // constant-density fluid, solid
  std::optional<double> specific_heat, conductivity;   // solid
};

struct InitialState {
  std::string kind;  ///< "fluid" or "solid"
  std::optional<double> density, pressure;
  std::optional<double> temperature;  ///< null for incompressible fluid zones
  std::optional<std::array<double, 3>> velocity;
  std::optional<double> nu_tilde, k, omega;
};

struct CellZoneInput {
  std::string zone;
  Medium medium;
  InitialState initial;
};

/// Per-face boundary states of an oracle (docs/contracts/face-states.md).
struct BoundaryStates {
  std::string path;    ///< absolute path of the autan.face-states/1 file
  std::string sha256;  ///< its expected SHA-256
};

/// The point vortex a farfield condition adds to the freestream (case-schema.md, Boundaries).
struct PointVortex {
  std::vector<std::string> zones;  ///< the wall or slip zones whose lift gives the circulation
  std::array<double, 3> center{};  ///< the vortex position (m)
};

/// The initial-state file of a case (docs/contracts/initial-state.md).
struct InitialStateFile {
  std::string path;    ///< absolute path of the autan.initial-state/1 file
  std::string sha256;  ///< its expected SHA-256
};

struct BoundaryInput {
  std::string zone;
  std::string kind;  ///< wall, slip, symmetry, farfield, inlet, velocity-inlet, outlet,
                     ///< manufactured, oracle-state
  std::optional<double> temperature, total_pressure, total_temperature, static_pressure;
  std::optional<double> velocity;           ///< velocity-inlet: speed normal to the face (m/s)
  std::optional<BoundaryStates> states;     ///< oracle-state only
  std::optional<PointVortex> point_vortex;  ///< farfield only
};

struct Turbulence {
  std::string model;  ///< inviscid, laminar, sa, sst, ddes, wmles
  std::optional<std::string> base, wall_model;
  std::optional<double> nu_tilde_ratio, intensity, viscosity_ratio;
};

struct Freestream {
  double density = 0.0, pressure = 0.0;
  std::optional<double> temperature;
  std::array<double, 3> velocity{};
  double speed = 0.0, viscosity = 0.0;
  std::optional<double> mach, re, ref_length;
  double aoa = 0.0, sideslip = 0.0;
  std::optional<double> nu_tilde, k, omega;
};

struct TimeInput {
  std::string kind;  ///< "steady" or "unsteady"
  std::int64_t max_iterations = 0;
  double residual_orders = 0.0;
  std::string linear_solver;  ///< steady: "multigrid" or "gauss-seidel" (global colours)
  std::int64_t steps = 0;
  std::optional<double> dt, cfl;
  std::optional<std::int64_t> average_from;
};

struct ForcesMonitor {
  std::string name;
  std::int64_t every = 1;
  std::vector<std::string> zones;
  std::optional<double> ref_area, ref_length;
  std::array<double, 3> moment_center{};
};

/// A manufactured solution (autan.mms/1) the case is verified against.
struct Verification {
  std::string path;    ///< absolute path of the autan.mms/1 file
  std::string sha256;  ///< its expected SHA-256
  std::string text;    ///< its bytes, read and checked by autan-run
};

/// The previous run a run starts from (solver-input.md, `start`).
struct Start {
  std::string run;  ///< absolute directory of the previous run
  std::string run_id, identity;
  std::string mode;  ///< "continue" or "warm"
  std::int64_t iterations = 0;
  std::optional<double> time;
  std::string solution_sha256;
  /// Residuals at iteration 1 of the original run, in residual_names order;
  /// set exactly for a steady `continue`.
  std::optional<std::vector<double>> residual_reference;
  /// Equations exempt at iteration 1 that left their floor earlier in the
  /// chain (case-schema.md#time), sorted; set exactly with residual_reference.
  std::optional<std::vector<std::string>> woken;
};

/// A periodic pair of the mesh, as MeshInfo lists it (solver-input.md,
/// `periodic`): zone a's faces map onto zone b's by the translation.
struct PeriodicInput {
  std::array<std::string, 2> zones;
  std::array<double, 3> translation{};
};

struct SolverInput {
  std::string case_name, case_sha256, case_toml;
  std::string mesh_path, mesh_sha256;
  std::string precision;  ///< "fp32+acc64" or "fp64"
  bool deterministic = false;
  std::string device_kind;  ///< cpu, cuda, hip
  std::int64_t device_count = 1;
  std::string physics_kind;  ///< compressible, incompressible
  Turbulence turbulence;
  Freestream freestream;
  std::vector<BoundaryInput> boundaries;
  std::vector<CellZoneInput> cell_zones;
  TimeInput time;
  std::vector<ForcesMonitor> monitors;
  /// Present for verification cases: every boundary is then "manufactured".
  std::optional<Verification> verification;
  std::optional<Start> start;
  std::optional<InitialStateFile> initial_state;  ///< never with start
  std::vector<PeriodicInput> periodic;
};

/// Parses and checks a solver input: every key present, no unknown key,
/// types, tags and the nulls the kinds require. Throws InputError (E0705)
/// naming the path of the first offending key.
SolverInput parse_solver_input(const std::string& json_text);

/// Equations of the history and fields of solution.cgns for this input, in
/// the order of docs/contracts/results.md.
std::vector<std::string> residual_names(const SolverInput& input);
std::vector<std::string> field_names(const SolverInput& input);

}  // namespace autan::runtime
