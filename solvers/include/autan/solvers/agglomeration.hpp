#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan::solvers {

/// One level of an agglomeration hierarchy: a graph whose nodes are the mesh's cells (level 0)
/// or agglomerates of the finer level's nodes, and whose edges are the faces between them.
/// Level 0's edges are the mesh's interior faces, in mesh order; a coarse level has one edge per
/// pair of adjacent nodes, owner < neighbour.
struct Level {
  std::size_t n_nodes = 0;
  std::vector<index_t> owner, neighbour;
  /// Lines of the smoother, CSR: a partition of the nodes into sequences in which consecutive
  /// nodes share an edge. Most lines hold one node.
  std::vector<offset_t> line_offsets{0};
  std::vector<index_t> line_nodes;
  /// Node of the next coarser level that holds each node; empty on the coarsest level.
  std::vector<index_t> coarse_node;
  /// Edge of the next coarser level that holds each edge, or -1 for an edge inside an
  /// agglomerate; `coarse_flipped` is 1 where the edge's owner lies in the coarse neighbour.
  std::vector<index_t> coarse_edge;
  std::vector<std::uint8_t> coarse_flipped;

  [[nodiscard]] std::size_t n_lines() const noexcept { return line_offsets.size() - 1; }
};

struct AgglomerationOptions {
  /// A face joins two cells of a line when its coupling A_f / |x_N - x_O| is at least this many
  /// times the weakest coupling of both cells.
  double line_ratio = 4.0;
  /// Most nodes of an isotropic agglomerate; a node left alone joins a neighbouring agglomerate
  /// beyond this size rather than stay alone.
  std::size_t max_agglomerate = 8;
  /// Most lines grouped in the wall plane.
  std::size_t max_lines = 4;
  /// Lines of fewer nodes are smoothed as lines but agglomerated as isotropic nodes (pairs of
  /// cells across a large face are not wall layers).
  std::size_t min_line = 4;
  /// Coarsening stops once a level has at most this many nodes, or when a level would keep more
  /// than `min_coarsening` of the finer level's nodes.
  std::size_t coarsest_nodes = 64;
  double min_coarsening = 0.8;
  int max_levels = 30;
};

/// The fine level's lines of at least two cells (Level::line_offsets / line_nodes of
/// agglomerate's level 0, single cells left out), CSR over the cells of `mesh`, each line in
/// its sweep order. Deterministic: the same mesh and geometry give the same lines, so every rank
/// of a run computes them alike on the global mesh. Throws std::invalid_argument on a mesh with
/// ghost cells.
struct Lines {
  std::vector<offset_t> offsets{0};
  std::vector<index_t> cells;
};
Lines smoother_lines(const Mesh& mesh, const Geometry& geometry,
                     const AgglomerationOptions& options = {});

/// Agglomeration hierarchy of a mesh without ghost cells, finest level first. Throws
/// std::invalid_argument on a mesh with ghost cells. Deterministic: the result depends only on
/// the mesh, its geometry and the options.
std::vector<Level> agglomerate(const Mesh& mesh, const Geometry& geometry,
                               const AgglomerationOptions& options = {});

}  // namespace autan::solvers
