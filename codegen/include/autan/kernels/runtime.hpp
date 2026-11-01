#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define AUTAN_HD __host__ __device__
#else
#define AUTAN_HD
#endif

namespace autan::kernels {

using index_t = std::int32_t;
using offset_t = std::int64_t;

/// Topology arrays that generated loops read; host or device pointers. Filled from
/// autan::MeshView (docs/contracts/mesh.md). Incidence k is entry k of the cell -> face CSR.
struct Topology {
  index_t n_cells;
  index_t n_owned_cells;
  index_t n_faces;
  index_t n_interior_faces;
  /// Faces that touch an owned cell (mesh contract, invariant 6): interior [0,
  /// n_owned_interior_faces), boundary [n_interior_faces, n_owned_faces). Face loops that
  /// increment cell data run over these only.
  index_t n_owned_interior_faces;
  index_t n_owned_faces;
  offset_t n_incidences;
  const index_t* owner;
  const index_t* neighbour;
  const offset_t* cell_face_offsets;
  const index_t* cell_faces;
  /// Incidence of the same interior face in the other cell's list (-1 on boundary faces), for
  /// the deterministic adjoint of cell reductions; null when the caller does not provide it.
  const offset_t* incidence_twin = nullptr;
  /// Cell stencil for reductions over=stencil (autan::kernels::cell_stencil): entries of cell c
  /// are stencil[stencil_offsets[c] .. stencil_offsets[c + 1]), a cell index, or -(f + 1) for a
  /// boundary face f; stencil_twin[k] is the entry of c in the stencil of cell stencil[k] (-1 for
  /// a boundary face). Null when the module has no stencil reduction.
  offset_t n_stencil = 0;
  const offset_t* stencil_offsets = nullptr;
  const index_t* stencil = nullptr;
  const offset_t* stencil_twin = nullptr;
  /// Sliced layout (autan::kernels::sliced): entry j of cell c is at stencil_offsets[c] +
  /// j * stencil_stride for j < stencil_length[c], so a warp reading entry j of 32 consecutive
  /// cells reads contiguous memory. stencil_stride == 1 is the plain CSR layout above.
  offset_t stencil_stride = 1;
  const index_t* stencil_length = nullptr;
  /// Faces whose local orientation is the reverse of their global one (partition faces,
  /// LocalPartition::face_reversed), or null. Face loops then evaluate such a face in its global
  /// orientation: left and right swapped, oriented geometry negated back, so a face gives the
  /// same bits on every rank count. Jacobian blocks follow: block "left" is the global left cell.
  const std::uint8_t* face_reversed = nullptr;
  /// Set with the global order (GlobalKeys): results must not depend on the partition, so face
  /// and boundary loops accept only the gather strategies (E1109), whose per-cell order is the
  /// global one; scatter, atomic and coloured orders follow the local numbering.
  bool rank_independent = false;
  /// Sliced incidence layout (autan::kernels::sliced_incidences): entry j of cell c is at
  /// cell_face_offsets[c] + j * incidence_stride for j < incidence_length[c], and n_incidences
  /// counts the padding; incidence_cell[k] is the cell across the incidence's face (-1 on a
  /// boundary face), so a reduction reads no owner or neighbour. Every per-incidence array
  /// (cell_faces, incidence_twin, incidence data) follows the layout. incidence_stride == 1 is
  /// the CSR layout, where incidence_cell may be null.
  offset_t incidence_stride = 1;
  const index_t* incidence_length = nullptr;
  const index_t* incidence_cell = nullptr;
};

/// Precision policies (docs/research/kernel-generator.md section 7): storage type of each role,
/// compute type of bodies, accumulation type of Inc targets and reductions.
struct fp64 {
  using state = double;
  using grad = double;
  using residual = double;
  using geom = double;
  using geom_value = double;  // Geom data that reach bodies only as values
  using param = double;
  using jac = double;
  using compute = double;
  using accum = double;
  static constexpr const char* name = "fp64";
};

struct fp32_acc64 {
  using state = float;
  using grad = float;
  using residual = double;
  using geom = double;
  using geom_value = float;  // Geom data that reach bodies only as values
  using param = double;
  using jac = float;
  using compute = float;
  using accum = double;
  static constexpr const char* name = "fp32+acc64";
};

struct fp32 {
  using state = float;
  using grad = float;
  using residual = float;
  using geom = float;
  using geom_value = float;  // Geom data that reach bodies only as values
  using param = float;
  using jac = float;
  using compute = float;
  using accum = float;
  static constexpr const char* name = "fp32";
};

/// Complex-step policy: compute in std::complex<double> on the host (tests only).
struct complex_step {
  using state = std::complex<double>;
  using grad = std::complex<double>;
  using residual = std::complex<double>;
  using geom = double;
  using geom_value = double;  // Geom data that reach bodies only as values
  using param = double;
  using jac = std::complex<double>;
  using compute = std::complex<double>;
  using accum = std::complex<double>;
  static constexpr const char* name = "complex-step";
};

// Comparisons go through these so that the complex-step policy compares real parts
// (Martins, Sturdza and Alonso 2003, section 4).
template <class T>
AUTAN_HD inline bool lt(T a, T b) {
  return a < b;
}
template <class T>
AUTAN_HD inline bool le(T a, T b) {
  return a <= b;
}
template <class T>
AUTAN_HD inline bool gt(T a, T b) {
  return a > b;
}
template <class T>
AUTAN_HD inline bool ge(T a, T b) {
  return a >= b;
}
template <class T>
inline bool lt(std::complex<T> a, std::complex<T> b) {
  return a.real() < b.real();
}
template <class T>
inline bool le(std::complex<T> a, std::complex<T> b) {
  return a.real() <= b.real();
}
template <class T>
inline bool gt(std::complex<T> a, std::complex<T> b) {
  return a.real() > b.real();
}
template <class T>
inline bool ge(std::complex<T> a, std::complex<T> b) {
  return a.real() >= b.real();
}

template <class T>
AUTAN_HD inline T infinity() {
  if constexpr (sizeof(T) == sizeof(float)) {
    return T(__builtin_huge_valf());
  } else {
    return T(__builtin_huge_val());
  }
}

}  // namespace autan::kernels
