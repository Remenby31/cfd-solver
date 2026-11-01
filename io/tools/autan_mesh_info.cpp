// autan-mesh-info: reads a mesh and prints its counts, zones and quality, or
// with --json the MeshInfo document of docs/contracts/mesh-info.md.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <numeric>
#include <string>
#include <string_view>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/quality.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"

namespace {

constexpr const char* usage =
    "usage: autan-mesh-info [--json] MESH\n"
    "\n"
    "Reads MESH (Gmsh .msh 4.1, Fluent .msh, or an OpenFOAM polyMesh or case\n"
    "directory) and prints its counts, zones and quality. With --json, prints\n"
    "the MeshInfo document (schema autan.mesh-info/1) instead. Exits 1 with a\n"
    "coded error (E01xx) if the mesh cannot be read, 2 on a usage error.\n";

double max_of(const std::vector<double>& v) {
  return v.empty() ? 0.0 : *std::ranges::max_element(v);
}

void print_text(const autan::Mesh& mesh, const std::string& hash) {
  const autan::Geometry g = autan::compute_geometry(mesh);
  const autan::MeshQuality q = autan::compute_quality(mesh, g);
  const auto closure = autan::cell_closure_error(mesh, g);
  std::printf("mesh      %s (%s)\nsha256    %s\n", mesh.source().path.c_str(),
              mesh.source().format.c_str(), hash.c_str());
  std::printf("cells     %d\nfaces     %d (%d interior)\nnodes     %d\nvolume    %.17g\n",
              mesh.n_cells(), mesh.n_faces(), mesh.n_interior_faces(), mesh.n_points(),
              std::accumulate(g.cell_volume.begin(), g.cell_volume.end(), 0.0));
  std::printf("\nboundary zones\n");
  const auto offsets = mesh.boundary_zone_offsets();
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    const auto& zone = mesh.face_zones()[z];
    if (!zone.boundary) continue;
    std::printf("  %-24s %-20s %d faces\n", zone.name.c_str(),
                zone.bc_type.empty() ? "-" : zone.bc_type.c_str(), offsets[z + 1] - offsets[z]);
  }
  std::printf("\ncell zones\n");
  for (std::size_t z = 0; z < mesh.cell_zones().size(); ++z) {
    const auto& zone = mesh.cell_zones()[z];
    const auto cells = std::count(mesh.cell_zone().begin(), mesh.cell_zone().end(),
                                  static_cast<autan::index_t>(z));
    std::printf("  %-24s %-20s %td cells\n", zone.name.c_str(),
                zone.type.empty() ? "-" : zone.type.c_str(), cells);
  }
  std::printf("\nquality\n");
  const double min_ratio = q.volume_ratio.empty() ? 1.0 : *std::ranges::min_element(q.volume_ratio);
  std::printf("  max non-orthogonality  %.3f deg\n", max_of(q.non_orthogonality));
  std::printf("  max skewness           %.3g\n", max_of(q.skewness));
  std::printf("  max aspect ratio       %.3g\n", max_of(q.aspect_ratio));
  std::printf("  min volume ratio       %.3g\n", min_ratio);
  std::printf("  max closure error      %.3g\n", max_of(closure));
}

}  // namespace

int main(int argc, char* const* argv) {
  bool json = false;
  const char* path = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "-h" || arg == "--help") {
      std::fputs(usage, stdout);
      return 0;
    } else if (path == nullptr && !arg.starts_with('-')) {
      path = argv[i];
    } else {
      std::fputs(usage, stderr);
      return 2;
    }
  }
  if (path == nullptr) {
    std::fputs(usage, stderr);
    return 2;
  }
  try {
    const autan::Mesh mesh = autan::io::read_mesh(path);
    const std::string hash = autan::io::mesh_hash(path);
    if (json) {
      const auto groups = autan::io::read_zone_groups(path, mesh, hash);
      std::printf("%s\n", autan::io::mesh_info_json(mesh, hash, groups).c_str());
    } else {
      print_text(mesh, hash);
    }
  } catch (const autan::Error& e) {
    std::fprintf(stderr, "autan-mesh-info: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "autan-mesh-info: internal error: %s\n", e.what());
    return 1;
  }
  return 0;
}
