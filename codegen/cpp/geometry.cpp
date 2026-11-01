#include "autan/kernels/geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace autan::kernels {

namespace {

using Vec = std::array<double, 3>;

Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

std::array<double, 9> inverse3(const std::array<double, 9>& m) {
  const double c00 = m[4] * m[8] - m[5] * m[7];
  const double c01 = m[5] * m[6] - m[3] * m[8];
  const double c02 = m[3] * m[7] - m[4] * m[6];
  const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  const double s = 1.0 / det;
  return {c00 * s, (m[2] * m[7] - m[1] * m[8]) * s, (m[1] * m[5] - m[2] * m[4]) * s,
          c01 * s, (m[0] * m[8] - m[2] * m[6]) * s, (m[2] * m[3] - m[0] * m[5]) * s,
          c02 * s, (m[1] * m[6] - m[0] * m[7]) * s, (m[0] * m[4] - m[1] * m[3]) * s};
}

/// Weighted least-squares weights w_k = M^-1 d_k / |d_k|^2, M = sum_k d_k d_k^T / |d_k|^2, of one
/// cell's entries, written to out[u * stride + first + k].
void store_lsq_weights(const std::vector<Vec>& d, std::vector<double>& out, std::size_t stride,
                       std::size_t first) {
  std::array<double, 9> mat{};
  for (const Vec& dk : d) {
    const double w = 1.0 / (dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
    for (std::size_t u = 0; u < 3; ++u) {
      for (std::size_t v = 0; v < 3; ++v) {
        mat[3 * u + v] += w * dk[u] * dk[v];
      }
    }
  }
  const auto inv = inverse3(mat);
  for (std::size_t j = 0; j < d.size(); ++j) {
    const Vec& dk = d[j];
    const double w = 1.0 / (dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
    for (std::size_t u = 0; u < 3; ++u) {
      out[u * stride + first + j] =
          w * (inv[3 * u] * dk[0] + inv[3 * u + 1] * dk[1] + inv[3 * u + 2] * dk[2]);
    }
  }
}

}  // namespace

Topology topology(const MeshView& mesh) noexcept {
  return {mesh.n_cells,
          mesh.n_owned_cells,
          mesh.n_faces,
          mesh.n_interior_faces,
          mesh.n_owned_interior_faces,
          mesh.n_owned_faces,
          mesh.cell_face_offsets[mesh.n_cells],
          mesh.owner,
          mesh.neighbour,
          mesh.cell_face_offsets,
          mesh.cell_faces,
          nullptr};
}

std::vector<offset_t> incidence_twins(const Mesh& mesh) {
  const auto off = mesh.cell_face_offsets();
  const auto cf = mesh.cell_faces();
  const auto own = mesh.owner();
  const auto nei = mesh.neighbour();
  std::vector<offset_t> twin(cf.size(), -1);
  for (std::size_t c = 0; c + 1 < off.size(); ++c) {
    for (auto k = off[c]; k < off[c + 1]; ++k) {
      const index_t f = cf[static_cast<std::size_t>(k)];
      const auto fi = static_cast<std::size_t>(f);
      if (fi >= nei.size()) {
        continue;
      }
      const auto other =
          static_cast<std::size_t>(static_cast<std::size_t>(own[fi]) == c ? nei[fi] : own[fi]);
      // Faces of a cell are listed in increasing order (mesh contract).
      const auto first = cf.begin() + off[other];
      const auto last = cf.begin() + off[other + 1];
      twin[static_cast<std::size_t>(k)] = std::lower_bound(first, last, f) - cf.begin();
    }
  }
  return twin;
}

GlobalOrder global_order(const Mesh& mesh, const GlobalKeys& keys) {
  const auto off = mesh.cell_face_offsets();
  const auto own = mesh.owner();
  const auto nei = mesh.neighbour();
  GlobalOrder out;
  out.cell_faces.assign(mesh.cell_faces().begin(), mesh.cell_faces().end());
  const std::size_t nc = off.size() - 1;
  auto by_key = [&](index_t a, index_t b) {
    const auto ka = keys.face[static_cast<std::size_t>(a)];
    const auto kb = keys.face[static_cast<std::size_t>(b)];
    return ka != kb ? ka < kb : a < b;
  };
  for (std::size_t c = 0; c < nc; ++c) {
    std::sort(out.cell_faces.begin() + off[c], out.cell_faces.begin() + off[c + 1], by_key);
  }
  out.twin.assign(out.cell_faces.size(), -1);
  for (std::size_t c = 0; c < nc; ++c) {
    for (auto k = off[c]; k < off[c + 1]; ++k) {
      const index_t f = out.cell_faces[static_cast<std::size_t>(k)];
      const auto fi = static_cast<std::size_t>(f);
      if (fi >= nei.size()) {
        continue;
      }
      const auto o =
          static_cast<std::size_t>(static_cast<std::size_t>(own[fi]) == c ? nei[fi] : own[fi]);
      const auto first = out.cell_faces.begin() + off[o];
      const auto last = out.cell_faces.begin() + off[o + 1];
      out.twin[static_cast<std::size_t>(k)] = std::find(first, last, f) - out.cell_faces.begin();
    }
  }
  return out;
}

PackedGeometry pack_geometry(const Mesh& mesh, const Geometry& g, const GlobalKeys* keys) {
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  PackedGeometry p;
  p.face_area.resize(3 * nf);
  p.face_centroid.resize(3 * nf);
  p.face_moment.assign(9 * nf, 0.0);
  p.cell_centroid.resize(3 * nc);
  p.cell_volume = g.cell_volume;
  for (std::size_t f = 0; f < nf; ++f) {
    p.face_area[f] = g.face_area_x[f];
    p.face_area[nf + f] = g.face_area_y[f];
    p.face_area[2 * nf + f] = g.face_area_z[f];
    p.face_centroid[f] = g.face_centroid_x[f];
    p.face_centroid[nf + f] = g.face_centroid_y[f];
    p.face_centroid[2 * nf + f] = g.face_centroid_z[f];
  }
  for (std::size_t c = 0; c < nc; ++c) {
    p.cell_centroid[c] = g.cell_centroid_x[c];
    p.cell_centroid[nc + c] = g.cell_centroid_y[c];
    p.cell_centroid[2 * nc + c] = g.cell_centroid_z[c];
  }
  p.face_area_magnitude = g.face_area;
  p.face_normal.resize(3 * nf);
  p.face_r_left.resize(3 * nf);
  p.face_r_right.assign(3 * nf, 0.0);
  {
    const auto own = mesh.owner();
    const auto nei = mesh.neighbour();
    for (std::size_t f = 0; f < nf; ++f) {
      const auto l = static_cast<std::size_t>(own[f]);
      for (std::size_t u = 0; u < 3; ++u) {
        p.face_normal[u * nf + f] = p.face_area[u * nf + f] / g.face_area[f];
        p.face_r_left[u * nf + f] = p.face_centroid[u * nf + f] - p.cell_centroid[u * nc + l];
        if (f < nei.size()) {
          const auto r = static_cast<std::size_t>(nei[f]);
          p.face_r_right[u * nf + f] = p.face_centroid[u * nf + f] - p.cell_centroid[u * nc + r];
        }
      }
    }
  }

  // Face moments over a fan of triangles (m, p_i, p_i+1), m the node mean. The fan starts at the
  // node of smallest (x, y, z) and runs in the face's global orientation (keys->reversed), and
  // the sum is signed for the local one: a face gives the same bits however it is numbered.
  const auto x = mesh.x(), y = mesh.y(), z = mesh.z();
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    const auto nodes = mesh.face_nodes(f);
    const std::size_t nn = nodes.size();
    const auto fi = static_cast<std::size_t>(f);
    auto point = [&](std::size_t i) {
      const auto n = static_cast<std::size_t>(nodes[i]);
      return Vec{x[n], y[n], z[n]};
    };
    std::size_t start = 0;
    for (std::size_t i = 1; i < nn; ++i) {
      if (point(i) < point(start)) {
        start = i;
      }
    }
    const bool reversed = keys != nullptr && !keys->reversed.empty() && keys->reversed[fi] != 0;
    auto at = [&](std::size_t i) {
      return point(reversed ? (start + nn - i % nn) % nn : (start + i) % nn);
    };
    Vec m{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < nn; ++i) {
      const Vec a = at(i);
      m = {m[0] + a[0], m[1] + a[1], m[2] + a[2]};
    }
    const auto k = static_cast<double>(nn);
    m = {m[0] / k, m[1] / k, m[2] / k};
    const Vec xf{g.face_centroid_x[fi], g.face_centroid_y[fi], g.face_centroid_z[fi]};
    std::array<double, 9> moment{};
    for (std::size_t i = 0; i < nn; ++i) {
      const Vec a = at(i);
      const Vec b = at(i + 1);
      const Vec s = cross(sub(a, m), sub(b, m));
      const Vec c{(m[0] + a[0] + b[0]) / 3.0, (m[1] + a[1] + b[1]) / 3.0,
                  (m[2] + a[2] + b[2]) / 3.0};
      const Vec r = sub(c, xf);
      for (std::size_t u = 0; u < 3; ++u) {
        for (std::size_t v = 0; v < 3; ++v) {
          moment[3 * u + v] += 0.5 * r[u] * s[v];
        }
      }
    }
    for (std::size_t j = 0; j < 9; ++j) {
      p.face_moment[j * nf + fi] = reversed ? -moment[j] : moment[j];
    }
  }

  const auto offsets = mesh.cell_face_offsets();
  // Incidences in the order the topology will list them: the mesh's, or the global one.
  const GlobalOrder order = keys != nullptr ? global_order(mesh, *keys) : GlobalOrder{};
  const std::span<const index_t> cell_faces =
      keys != nullptr ? std::span<const index_t>(order.cell_faces) : mesh.cell_faces();
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  const auto ni = static_cast<std::size_t>(offsets.back());
  p.lsq_weight.assign(3 * ni, 0.0);
  p.incidence_side.assign(ni, 1.0);
  for (std::size_t c = 0; c + 1 < offsets.size(); ++c) {
    for (auto k = offsets[c]; k < offsets[c + 1]; ++k) {
      const auto f = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(k)]);
      // +1 for the face's left cell in its global orientation (keys->reversed).
      const bool reversed = keys != nullptr && !keys->reversed.empty() && keys->reversed[f] != 0;
      if (f < neighbour.size() && (static_cast<std::size_t>(owner[f]) != c) != reversed) {
        p.incidence_side[static_cast<std::size_t>(k)] = -1.0;
      }
    }
  }
  std::vector<Vec> d;
  for (index_t c = 0; c < mesh.n_owned_cells(); ++c) {
    const auto ci = static_cast<std::size_t>(c);
    const Vec xc{p.cell_centroid[ci], p.cell_centroid[nc + ci], p.cell_centroid[2 * nc + ci]};
    d.clear();
    for (auto k = offsets[ci]; k < offsets[ci + 1]; ++k) {
      const auto f = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(k)]);
      Vec xo;
      if (f < neighbour.size()) {
        const auto o = static_cast<std::size_t>(owner[f] == c ? neighbour[f] : owner[f]);
        xo = {p.cell_centroid[o], p.cell_centroid[nc + o], p.cell_centroid[2 * nc + o]};
      } else {
        xo = {p.face_centroid[f], p.face_centroid[nf + f], p.face_centroid[2 * nf + f]};
      }
      d.push_back(sub(xo, xc));
    }
    store_lsq_weights(d, p.lsq_weight, ni, static_cast<std::size_t>(offsets[ci]));
  }
  return p;
}

namespace {

// Rows of a CSR (n + 1 offsets) in slices of `slice` rows: starts (n + 1, the last the total
// size), lengths, and the sliced position of each CSR entry.
struct Slices {
  std::vector<offset_t> starts;
  std::vector<index_t> length;
  std::vector<offset_t> position;
};

Slices slice_rows(std::span<const offset_t> offsets, int slice) {
  if (slice < 1) {
    throw std::invalid_argument("E1101: slice must be at least 1");
  }
  const std::size_t nc = offsets.size() - 1;
  const auto w = static_cast<std::size_t>(slice);
  Slices out;
  out.length.resize(nc);
  out.position.resize(static_cast<std::size_t>(offsets[nc]));
  out.starts.resize(nc + 1);
  offset_t start = 0;
  for (std::size_t s = 0; s * w < nc; ++s) {
    offset_t longest = 0;
    for (std::size_t l = 0; l < w && s * w + l < nc; ++l) {
      const std::size_t c = s * w + l;
      longest = std::max(longest, offsets[c + 1] - offsets[c]);
    }
    for (std::size_t l = 0; l < w && s * w + l < nc; ++l) {
      const std::size_t c = s * w + l;
      out.starts[c] = start + static_cast<offset_t>(l);
      out.length[c] = static_cast<index_t>(offsets[c + 1] - offsets[c]);
      for (offset_t j = 0; j < out.length[c]; ++j) {
        out.position[static_cast<std::size_t>(offsets[c] + j)] =
            out.starts[c] + j * static_cast<offset_t>(slice);
      }
    }
    start += longest * static_cast<offset_t>(slice);
  }
  out.starts[nc] = start;
  return out;
}

}  // namespace

SlicedIncidences sliced_incidences(const Topology& csr, int slice) {
  const auto nc = static_cast<std::size_t>(csr.n_cells);
  auto rows = slice_rows(std::span(csr.cell_face_offsets, nc + 1), slice);
  SlicedIncidences out;
  out.stride = slice;
  const auto n = static_cast<std::size_t>(rows.starts[nc]);
  out.cell_faces.assign(n, 0);
  out.cell.assign(n, -1);
  out.twin.assign(n, -1);
  for (std::size_t c = 0; c < nc; ++c) {
    for (auto k = csr.cell_face_offsets[c]; k < csr.cell_face_offsets[c + 1]; ++k) {
      const auto p = static_cast<std::size_t>(rows.position[static_cast<std::size_t>(k)]);
      const index_t f = csr.cell_faces[k];
      out.cell_faces[p] = f;
      if (f < csr.n_interior_faces) {
        out.cell[p] = csr.owner[f] == static_cast<index_t>(c) ? csr.neighbour[f] : csr.owner[f];
      }
      const offset_t t = csr.incidence_twin == nullptr ? -1 : csr.incidence_twin[k];
      out.twin[p] = t < 0 ? -1 : rows.position[static_cast<std::size_t>(t)];
    }
  }
  out.cell_face_offsets = std::move(rows.starts);
  out.length = std::move(rows.length);
  out.position = std::move(rows.position);
  return out;
}

std::vector<double> sliced_values(const SlicedIncidences& s, const std::vector<double>& csr,
                                  int comps) {
  const std::size_t ne = s.position.size();
  const std::size_t n = s.cell_faces.size();
  const auto nk = static_cast<std::size_t>(comps);
  if (csr.size() != nk * ne) {
    throw std::invalid_argument("E1101: per-incidence values of the wrong size");
  }
  std::vector<double> out(nk * n, 0.0);
  for (std::size_t k = 0; k < ne; ++k) {
    for (std::size_t i = 0; i < nk; ++i) {
      out[i * n + static_cast<std::size_t>(s.position[k])] = csr[i * ne + k];
    }
  }
  return out;
}

SlicedStencil sliced(const CellStencil& csr, int slice) {
  auto rows = slice_rows(csr.offsets, slice);
  SlicedStencil out;
  out.stride = slice;
  out.stencil.offsets = std::move(rows.starts);
  out.length = std::move(rows.length);
  out.position = std::move(rows.position);
  const std::size_t nc = csr.offsets.size() - 1;
  const auto n = static_cast<std::size_t>(out.stencil.offsets[nc]);
  const std::size_t ne = csr.entries.size();
  out.stencil.entries.assign(n, 0);
  out.stencil.twin.assign(n, -1);
  out.stencil.lsq_weight.assign(3 * n, 0.0);
  for (std::size_t k = 0; k < ne; ++k) {
    const auto p = static_cast<std::size_t>(out.position[k]);
    out.stencil.entries[p] = csr.entries[k];
    const offset_t t = csr.twin.empty() ? -1 : csr.twin[k];
    out.stencil.twin[p] = t < 0 ? -1 : out.position[static_cast<std::size_t>(t)];
    if (!csr.lsq_weight.empty()) {
      for (std::size_t i = 0; i < 3; ++i) {
        out.stencil.lsq_weight[i * n + p] = csr.lsq_weight[i * ne + k];
      }
    }
  }
  return out;
}

CellStencil cell_stencil(const Mesh& mesh, const PackedGeometry& g, StencilKind kind,
                         const GlobalKeys* keys) {
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto n_owned = static_cast<std::size_t>(mesh.n_owned_cells());
  const auto off = mesh.cell_face_offsets();
  const auto cf = mesh.cell_faces();
  const auto own = mesh.owner();
  const auto nei = mesh.neighbour();
  const std::size_t nint = nei.size();
  auto faces_of = [&](std::size_t c) {
    return cf.subspan(static_cast<std::size_t>(off[c]),
                      static_cast<std::size_t>(off[c + 1] - off[c]));
  };
  auto across = [&](std::size_t c, std::size_t f) {
    return static_cast<std::size_t>(static_cast<std::size_t>(own[f]) == c ? nei[f] : own[f]);
  };

  // Cells and boundary faces touching each node (vertex stencil).
  std::vector<std::vector<index_t>> node_cells, node_bfaces;
  if (kind == StencilKind::vertex) {
    node_cells.resize(static_cast<std::size_t>(mesh.n_points()));
    node_bfaces.resize(node_cells.size());
    for (std::size_t f = 0; f < nf; ++f) {
      for (const index_t n : mesh.face_nodes(static_cast<index_t>(f))) {
        auto& cells = node_cells[static_cast<std::size_t>(n)];
        cells.push_back(own[f]);
        if (f < nint) {
          cells.push_back(nei[f]);
        } else {
          node_bfaces[static_cast<std::size_t>(n)].push_back(static_cast<index_t>(f));
        }
      }
    }
  }

  CellStencil st;
  st.offsets.assign(nc + 1, 0);
  std::vector<index_t> cells, bfaces;
  for (std::size_t c = 0; c < nc; ++c) {
    cells.clear();
    bfaces.clear();
    if (kind == StencilKind::vertex) {
      for (const index_t f : faces_of(c)) {
        for (const index_t n : mesh.face_nodes(f)) {
          const auto& around = node_cells[static_cast<std::size_t>(n)];
          cells.insert(cells.end(), around.begin(), around.end());
          const auto& bound = node_bfaces[static_cast<std::size_t>(n)];
          bfaces.insert(bfaces.end(), bound.begin(), bound.end());
        }
      }
    } else {
      for (const index_t f : faces_of(c)) {
        const auto fi = static_cast<std::size_t>(f);
        if (fi >= nint) {
          bfaces.push_back(f);
          continue;
        }
        const std::size_t n = across(c, fi);
        cells.push_back(static_cast<index_t>(n));
        for (const index_t f2 : faces_of(n)) {
          const auto f2i = static_cast<std::size_t>(f2);
          if (f2i >= nint) {
            bfaces.push_back(f2);
          } else {
            cells.push_back(static_cast<index_t>(across(n, f2i)));
          }
        }
      }
    }
    // Cells by (global id, translation), else index; then boundary faces by global id, else
    // index. A periodic image shares its cell's global id: its translation tells them apart.
    auto by_cell = [&](index_t a, index_t b) {
      if (keys == nullptr || keys->cell.empty()) {
        return a < b;
      }
      auto key = [&](index_t i) {
        std::array<double, 3> t{};
        const auto owned = static_cast<index_t>(n_owned);
        if (i >= owned && !keys->shift.empty()) {
          t = keys->shift[static_cast<std::size_t>(i - owned)];
        }
        return std::make_pair(keys->cell[static_cast<std::size_t>(i)], t);
      };
      const auto ka = key(a);
      const auto kb = key(b);
      return ka != kb ? ka < kb : a < b;
    };
    auto by = [](std::span<const std::int64_t> key) {
      return [key](index_t a, index_t b) {
        if (key.empty()) {
          return a < b;
        }
        const auto ka = key[static_cast<std::size_t>(a)];
        const auto kb = key[static_cast<std::size_t>(b)];
        return ka != kb ? ka < kb : a < b;
      };
    };
    std::sort(cells.begin(), cells.end(), by_cell);
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    cells.erase(std::remove(cells.begin(), cells.end(), static_cast<index_t>(c)), cells.end());
    std::sort(bfaces.begin(), bfaces.end(),
              by(keys != nullptr ? keys->face : std::span<const std::int64_t>{}));
    bfaces.erase(std::unique(bfaces.begin(), bfaces.end()), bfaces.end());
    st.entries.insert(st.entries.end(), cells.begin(), cells.end());
    for (const index_t f : bfaces) {
      st.entries.push_back(-f - 1);
    }
    st.offsets[c + 1] = static_cast<offset_t>(st.entries.size());
  }

  const std::size_t ns = st.entries.size();
  st.twin.assign(ns, -1);
  for (std::size_t c = 0; c < nc; ++c) {
    for (auto k = st.offsets[c]; k < st.offsets[c + 1]; ++k) {
      const index_t e = st.entries[static_cast<std::size_t>(k)];
      if (e < 0) {
        continue;
      }
      const auto n = static_cast<std::size_t>(e);
      const auto first = st.entries.begin() + st.offsets[n];
      const auto last = st.entries.begin() + st.offsets[n + 1];
      const auto cells_end = std::find_if(first, last, [](index_t x) { return x < 0; });
      const auto it = std::find(first, cells_end, static_cast<index_t>(c));
      if (it != cells_end && *it == static_cast<index_t>(c)) {
        st.twin[static_cast<std::size_t>(k)] = it - st.entries.begin();
      }
    }
  }

  st.lsq_weight.assign(3 * ns, 0.0);
  std::vector<Vec> d;
  for (std::size_t c = 0; c < n_owned; ++c) {
    const Vec xc{g.cell_centroid[c], g.cell_centroid[nc + c], g.cell_centroid[2 * nc + c]};
    d.clear();
    for (auto k = st.offsets[c]; k < st.offsets[c + 1]; ++k) {
      const index_t e = st.entries[static_cast<std::size_t>(k)];
      Vec xo;
      if (e >= 0) {
        const auto o = static_cast<std::size_t>(e);
        xo = {g.cell_centroid[o], g.cell_centroid[nc + o], g.cell_centroid[2 * nc + o]};
      } else {
        const auto f = static_cast<std::size_t>(-e - 1);
        xo = {g.face_centroid[f], g.face_centroid[nf + f], g.face_centroid[2 * nf + f]};
      }
      d.push_back(sub(xo, xc));
    }
    store_lsq_weights(d, st.lsq_weight, ns, static_cast<std::size_t>(st.offsets[c]));
  }
  return st;
}

}  // namespace autan::kernels
