#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Mesh hash recorded by every run, as 64 lowercase hex characters: SHA-256
/// of the file bytes, or for a polyMesh directory of the concatenation, over
/// points, faces, owner, neighbour, boundary and cellZones (when present), of
/// "<name>\n<byte length>\n<bytes>" (docs/contracts/mesh-info.md).
std::string mesh_hash(const std::filesystem::path& path);

/// Zones in MeshInfo order: boundary zones are the face zones with
/// `boundary` set, cell zones those holding at least one cell; indices into
/// Mesh::face_zones() and Mesh::cell_zones().
struct MeshInfoZones {
  std::vector<index_t> boundary;
  std::vector<index_t> cell;
};
MeshInfoZones meshinfo_zones(const Mesh& mesh);

/// Groups of boundary zones from a mesh's sidecar (docs/contracts/mesh-info.md,
/// Zone groups): each name with its zones, sorted by name, zones in MeshInfo
/// order; `path` the sidecar's, symlinks of the mesh path followed, `sha256` of its bytes.
struct ZoneGroups {
  std::string path, sha256;
  std::vector<std::pair<std::string, std::string>> provenance;
  std::vector<std::pair<std::string, std::vector<std::string>>> groups;
};

/// The sidecar of the mesh at `mesh_path` (`<mesh file>.zones.json`, for a
/// polyMesh directory `<directory>.zones.json`, symlinks followed), checked against `mesh` and
/// its hash `sha256`; nullopt when there is none. Throws E0136 for a sidecar
/// that cannot be read or breaks autan.zone-groups/1, E0137 for another mesh's
/// hash, E0138 for an unknown zone, an empty group, a bad name or a zone twice
/// in a group, E0139 for a name given twice.
std::optional<ZoneGroups> read_zone_groups(const std::filesystem::path& mesh_path, const Mesh& mesh,
                                           const std::string& sha256);

/// MeshInfo document (docs/contracts/mesh-info.md, schema autan.mesh-info/1)
/// of an assembled, unpartitioned mesh read from its source, as JSON text,
/// with the zone groups of its sidecar when it has one.
std::string mesh_info_json(const Mesh& mesh, const std::string& sha256,
                           const std::optional<ZoneGroups>& groups = std::nullopt);

}  // namespace autan::io
