#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace autan {

// Codes are stable identifiers shared with the SDK error registry; the number
// is printed as E#### (see docs/contracts/mesh.md, "Errors").
enum class ErrorCode : std::uint16_t {
  mesh_inconsistent_sizes = 101,
  mesh_index_out_of_range = 102,
  mesh_face_too_few_nodes = 103,
  mesh_face_repeated_node = 104,
  mesh_face_same_cell_both_sides = 105,
  mesh_cell_too_few_faces = 106,
  mesh_cell_not_closed = 107,
  mesh_face_shared_by_more_than_two_cells = 108,
  mesh_face_zone_role = 109,
  mesh_duplicate_zone_name = 110,
  mesh_labelled_face_not_found = 111,
  mesh_empty = 112,
  mesh_too_large = 113,
  mesh_shared_face_orientation = 114,
  mesh_face_labelled_twice = 115,
  mesh_duplicate_global_id = 117,
  mesh_periodic_invalid = 118,
  mesh_periodic_unmatched = 119,
  geometry_degenerate_face = 120,
  geometry_non_positive_volume = 121,
  io_cannot_open = 130,
  io_unknown_format = 131,
  io_malformed = 132,
  io_unsupported = 133,
  io_external_reference = 134,
  io_ambiguous_zone = 135,
  zone_groups_invalid = 136,
  zone_groups_other_mesh = 137,
  zone_groups_bad_group = 138,
  zone_groups_duplicate_name = 139,
  partition_invalid_request = 140,
  partition_input_has_ghosts = 141,
  partition_backend_failed = 142,
  partition_empty_part = 143,
  halo_inconsistent = 144,
  halo_device_unsupported = 145,
  mpi_unavailable = 146,
};

/// Returns the registry form of a code, for example "E0107".
std::string error_code_string(ErrorCode code);

/// Exception thrown by every autan library entry point. Callers branch on
/// code(); what() is a human-readable message whose wording is not part of the
/// contract.
class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, std::string_view message);
  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

}  // namespace autan
