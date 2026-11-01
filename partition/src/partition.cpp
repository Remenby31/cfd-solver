#include "autan/partition/partition.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "autan/core/error.hpp"
#include "groups.hpp"

#ifdef AUTAN_HAVE_METIS
#include <metis.h>
#endif
#ifdef AUTAN_HAVE_SCOTCH
#include <cstdio>

#include <scotch.h>
#endif

namespace autan {
namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

/// Cell adjacency through interior faces, CSR, duplicates merged (two cells
/// may share several faces).
struct DualGraph {
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> adjacency;
};

DualGraph dual_graph(const Mesh& mesh) {
  const auto n_cells = at(mesh.n_cells());
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  std::vector<std::int64_t> degree(n_cells + 1, 0);
  // Interior faces are sorted by (owner, neighbour): repeated pairs are adjacent.
  for (std::size_t f = 0; f < neighbour.size(); ++f) {
    if (f > 0 && owner[f] == owner[f - 1] && neighbour[f] == neighbour[f - 1]) continue;
    ++degree[at(owner[f]) + 1];
    ++degree[at(neighbour[f]) + 1];
  }
  DualGraph g;
  g.offsets.resize(n_cells + 1);
  g.offsets[0] = 0;
  for (std::size_t c = 0; c < n_cells; ++c) g.offsets[c + 1] = g.offsets[c] + degree[c + 1];
  g.adjacency.resize(at(g.offsets[n_cells]));
  std::vector<std::int64_t> fill(g.offsets.begin(), g.offsets.end() - 1);
  for (std::size_t f = 0; f < neighbour.size(); ++f) {
    if (f > 0 && owner[f] == owner[f - 1] && neighbour[f] == neighbour[f - 1]) continue;
    g.adjacency[at(fill[at(owner[f])]++)] = neighbour[f];
    g.adjacency[at(fill[at(neighbour[f])]++)] = owner[f];
  }
  return g;
}

// The dual graph with the cells of each node of `node_of` contracted into one.
// Throws E0140 for fewer nodes than parts.
DualGraph contracted(const DualGraph& graph, std::span<const std::int64_t> node_of,
                     std::int32_t n_parts) {
  const std::size_t n_cells = node_of.size();
  const std::int64_t n_nodes = *std::ranges::max_element(node_of) + 1;
  if (n_nodes < n_parts) {
    throw Error(ErrorCode::partition_invalid_request, "cannot cut " + std::to_string(n_nodes) +
                                                          " groups and cells into " +
                                                          std::to_string(n_parts) + " parts");
  }
  std::vector<std::vector<std::int64_t>> around(at(n_nodes));
  for (std::size_t c = 0; c < n_cells; ++c) {
    for (auto k = graph.offsets[c]; k < graph.offsets[c + 1]; ++k) {
      const std::int64_t a = node_of[c], b = node_of[at(graph.adjacency[at(k)])];
      if (a != b) around[at(a)].push_back(b);
    }
  }
  DualGraph out;
  out.offsets.assign(1, 0);
  for (auto& list : around) {
    std::ranges::sort(list);
    const auto end = std::unique(list.begin(), list.end());
    out.adjacency.insert(out.adjacency.end(), list.begin(), end);
    out.offsets.push_back(static_cast<std::int64_t>(out.adjacency.size()));
  }
  return out;
}

void check_unpartitioned(const Mesh& mesh) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw Error(ErrorCode::partition_input_has_ghosts,
                "the mesh to partition has " +
                    std::to_string(mesh.n_cells() - mesh.n_owned_cells()) + " ghost cells");
  }
}

void check_weights(const Mesh& mesh, std::span<const std::int32_t> weights) {
  if (weights.empty()) return;
  if (weights.size() != at(mesh.n_cells())) {
    throw Error(ErrorCode::partition_invalid_request,
                "cell_weights has " + std::to_string(weights.size()) + " entries for " +
                    std::to_string(mesh.n_cells()) + " cells");
  }
  const auto low = std::min_element(weights.begin(), weights.end());
  if (*low < 1) {
    throw Error(ErrorCode::partition_invalid_request,
                "cell weight " + std::to_string(*low) + " of cell " +
                    std::to_string(low - weights.begin()) + " is below 1");
  }
}

/// Throws E0142 unless the graph's arc count and total vertex weight fit the
/// backend's integer type `Int`, which also accumulates the weights.
template <class Int>
void check_fits(const DualGraph& graph, std::span<const std::int32_t> weights, const char* name) {
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<Int>::max());
  std::uint64_t total = 0;
  for (const std::int32_t w : weights) total += static_cast<std::uint64_t>(w);
  if (weights.empty()) total = graph.offsets.size() - 1;
  if (graph.adjacency.size() > limit || total > limit) {
    throw Error(ErrorCode::partition_backend_failed,
                std::string("the dual graph (") + std::to_string(graph.adjacency.size()) +
                    " arcs, total weight " + std::to_string(total) + ") exceeds the " +
                    std::to_string(8 * sizeof(Int)) + "-bit integers of this " + name + " build");
  }
}

#ifdef AUTAN_HAVE_METIS
std::vector<std::int32_t> partition_metis(const DualGraph& graph, std::int32_t n_parts,
                                          const PartitionOptions& options) {
  check_fits<idx_t>(graph, options.cell_weights, "METIS");
  idx_t n_vertices = static_cast<idx_t>(graph.offsets.size() - 1);
  idx_t n_constraints = 1;
  idx_t n = n_parts;
  std::vector<idx_t> xadj(graph.offsets.begin(), graph.offsets.end());
  std::vector<idx_t> adjncy(graph.adjacency.begin(), graph.adjacency.end());
  std::vector<idx_t> vwgt(options.cell_weights.begin(), options.cell_weights.end());
  real_t ubvec = static_cast<real_t>(1.0 + options.imbalance_tolerance);
  idx_t metis_options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(metis_options);
  metis_options[METIS_OPTION_SEED] = options.seed;
  metis_options[METIS_OPTION_NUMBERING] = 0;
  idx_t edge_cut = 0;
  std::vector<idx_t> part(static_cast<std::size_t>(n_vertices));
  const int status = METIS_PartGraphKway(&n_vertices, &n_constraints, xadj.data(), adjncy.data(),
                                         vwgt.empty() ? nullptr : vwgt.data(), nullptr, nullptr, &n,
                                         nullptr, &ubvec, metis_options, &edge_cut, part.data());
  if (status != METIS_OK) {
    throw Error(ErrorCode::partition_backend_failed,
                "METIS_PartGraphKway returned " + std::to_string(status));
  }
  return {part.begin(), part.end()};
}
#endif

#ifdef AUTAN_HAVE_SCOTCH
std::vector<std::int32_t> partition_scotch(const DualGraph& graph, std::int32_t n_parts,
                                           const PartitionOptions& options) {
  check_fits<SCOTCH_Num>(graph, options.cell_weights, "Scotch");
  const auto n_vertices = static_cast<SCOTCH_Num>(graph.offsets.size() - 1);
  std::vector<SCOTCH_Num> verttab(graph.offsets.begin(), graph.offsets.end());
  std::vector<SCOTCH_Num> edgetab(graph.adjacency.begin(), graph.adjacency.end());
  std::vector<SCOTCH_Num> velotab(options.cell_weights.begin(), options.cell_weights.end());
  std::vector<SCOTCH_Num> part(static_cast<std::size_t>(n_vertices));

  // A private, seeded generator and one thread make the result reproducible (Scotch 7 user's
  // guide, "Execution contexts"). SCOTCH_OPTIONNUMDETERMINISTIC alone is not enough: with 8 or
  // more threads the same call returned different parts on GB10, and a thread count taken from
  // the machine would give different parts on different machines.
  SCOTCH_Context context;
  SCOTCH_Graph graph_s;
  SCOTCH_Graph bound;
  SCOTCH_Strat strategy;
  if (SCOTCH_contextInit(&context) != 0 || SCOTCH_graphInit(&graph_s) != 0 ||
      SCOTCH_graphInit(&bound) != 0 || SCOTCH_stratInit(&strategy) != 0) {
    throw Error(ErrorCode::partition_backend_failed, "Scotch initialisation failed");
  }
  int status = SCOTCH_contextThreadSpawn(&context, 1, nullptr);
  if (status == 0) status = SCOTCH_contextOptionSetNum(&context, SCOTCH_OPTIONNUMDETERMINISTIC, 1);
  if (status == 0) status = SCOTCH_contextRandomClone(&context);
  if (status == 0) {
    SCOTCH_contextRandomSeed(&context, options.seed);
    status = SCOTCH_graphBuild(&graph_s, 0, n_vertices, verttab.data(), nullptr,
                               velotab.empty() ? nullptr : velotab.data(), nullptr,
                               static_cast<SCOTCH_Num>(edgetab.size()), edgetab.data(), nullptr);
  }
  if (status == 0) status = SCOTCH_contextBindGraph(&context, &graph_s, &bound);
  if (status == 0) {
    status = SCOTCH_stratGraphMapBuild(&strategy, SCOTCH_STRATQUALITY, n_parts,
                                       options.imbalance_tolerance);
  }
  if (status == 0) status = SCOTCH_graphPart(&bound, n_parts, &strategy, part.data());
  SCOTCH_stratExit(&strategy);
  SCOTCH_graphExit(&bound);
  SCOTCH_graphExit(&graph_s);
  SCOTCH_contextExit(&context);
  if (status != 0) {
    throw Error(ErrorCode::partition_backend_failed,
                "Scotch partitioning returned " + std::to_string(status));
  }
  std::vector<std::int32_t> result(part.size());
  std::transform(part.begin(), part.end(), result.begin(),
                 [](SCOTCH_Num p) { return static_cast<std::int32_t>(p); });
  return result;
}
#endif

}  // namespace

// Each cell's node once the groups of `options` contract: groups first in
// their order, then the other cells in theirs; empty when there is no group.
// Throws E0140 for malformed offsets, an empty group, or a cell out of range
// or in two groups.
std::vector<std::int64_t> detail::group_nodes(const Mesh& mesh, std::span<const offset_t> offsets,
                                              std::span<const index_t> group_cells) {
  if (offsets.empty()) {
    if (!group_cells.empty()) {
      throw Error(ErrorCode::partition_invalid_request, "group_cells without group_offsets");
    }
    return {};
  }
  if (offsets.front() != 0 || at(offsets.back()) != group_cells.size() ||
      std::ranges::adjacent_find(offsets, std::greater_equal{}) != offsets.end()) {
    throw Error(ErrorCode::partition_invalid_request,
                "group offsets are malformed or a group is empty");
  }
  std::vector<std::int64_t> node_of(at(mesh.n_cells()), -1);
  const std::size_t n_groups = offsets.size() - 1;
  for (std::size_t g = 0; g < n_groups; ++g) {
    for (auto k = offsets[g]; k < offsets[g + 1]; ++k) {
      const index_t c = group_cells[at(k)];
      if (c < 0 || c >= mesh.n_cells() || node_of[at(c)] != -1) {
        throw Error(ErrorCode::partition_invalid_request,
                    "cell " + std::to_string(c) + " is out of range or in two groups");
      }
      node_of[at(c)] = static_cast<std::int64_t>(g);
    }
  }
  auto next = static_cast<std::int64_t>(n_groups);
  for (auto& node : node_of) {
    if (node == -1) node = next++;
  }
  return node_of;
}

std::vector<std::int32_t> cell_cost_weights(const Mesh& mesh) {
  const auto offsets = mesh.cell_face_offsets();
  std::vector<std::int32_t> weights(at(mesh.n_cells()));
  for (std::size_t c = 0; c < weights.size(); ++c) {
    weights[c] = static_cast<std::int32_t>(offsets[c + 1] - offsets[c]);
  }
  return weights;
}

std::vector<std::int32_t> partition_cells(const Mesh& mesh, std::int32_t n_parts,
                                          const PartitionOptions& options) {
  check_unpartitioned(mesh);
  if (n_parts < 1 || n_parts > mesh.n_cells()) {
    throw Error(ErrorCode::partition_invalid_request,
                "cannot cut " + std::to_string(mesh.n_cells()) + " cells into " +
                    std::to_string(n_parts) + " parts");
  }
  check_weights(mesh, options.cell_weights);
  if (!(options.imbalance_tolerance >= 0.0)) {
    throw Error(ErrorCode::partition_invalid_request, "imbalance_tolerance must be ≥ 0");
  }
  // Groups contract into one node each; the other cells stay alone.
  const std::vector<std::int64_t> node_of =
      detail::group_nodes(mesh, options.group_offsets, options.group_cells);
  if (n_parts == 1) return std::vector<std::int32_t>(at(mesh.n_cells()), 0);
  const DualGraph graph =
      node_of.empty() ? dual_graph(mesh) : contracted(dual_graph(mesh), node_of, n_parts);
  PartitionOptions contracted_options = options;
  std::vector<std::int32_t> weights;
  if (!node_of.empty()) {
    weights.assign(graph.offsets.size() - 1, 0);
    for (std::size_t c = 0; c < node_of.size(); ++c) {
      const std::int64_t w = options.cell_weights.empty() ? 1 : options.cell_weights[c];
      const std::int64_t sum = weights[at(node_of[c])] + w;
      if (sum > std::numeric_limits<std::int32_t>::max()) {
        throw Error(ErrorCode::partition_invalid_request, "a group weighs more than 2^31 - 1");
      }
      weights[at(node_of[c])] = static_cast<std::int32_t>(sum);
    }
    contracted_options.cell_weights = weights;
  }
  std::vector<std::int32_t> parts;
  [[maybe_unused]] const PartitionOptions& method_options = contracted_options;
  switch (options.method) {
    case PartitionMethod::metis:
#ifdef AUTAN_HAVE_METIS
      parts = partition_metis(graph, n_parts, method_options);
      break;
#else
      throw Error(ErrorCode::partition_backend_failed,
                  "this build has no METIS (configure with AUTAN_ENABLE_METIS=ON)");
#endif
    case PartitionMethod::scotch:
#ifdef AUTAN_HAVE_SCOTCH
      parts = partition_scotch(graph, n_parts, method_options);
      break;
#else
      throw Error(ErrorCode::partition_backend_failed,
                  "this build has no Scotch (configure with AUTAN_ENABLE_SCOTCH=ON)");
#endif
  }

  if (!node_of.empty()) {
    std::vector<std::int32_t> of_cells(node_of.size());
    for (std::size_t c = 0; c < node_of.size(); ++c) of_cells[c] = parts[at(node_of[c])];
    parts = std::move(of_cells);
  }
  std::vector<std::int64_t> count(at(n_parts), 0);
  for (const std::int32_t p : parts) ++count[at(p)];
  const auto empty = std::find(count.begin(), count.end(), 0);
  if (empty != count.end()) {
    throw Error(ErrorCode::partition_empty_part, "part " + std::to_string(empty - count.begin()) +
                                                     " of " + std::to_string(n_parts) +
                                                     " received no cell");
  }
  return parts;
}

PartitionQuality partition_quality(const Mesh& mesh, std::span<const std::int32_t> parts,
                                   std::int32_t n_parts,
                                   std::span<const std::int32_t> cell_weights) {
  check_unpartitioned(mesh);
  check_weights(mesh, cell_weights);
  if (n_parts < 1 || parts.size() != at(mesh.n_cells())) {
    throw Error(ErrorCode::partition_invalid_request,
                "parts has " + std::to_string(parts.size()) + " entries for " +
                    std::to_string(mesh.n_cells()) + " cells and " + std::to_string(n_parts) +
                    " parts");
  }
  for (std::size_t c = 0; c < parts.size(); ++c) {
    if (parts[c] < 0 || parts[c] >= n_parts) {
      throw Error(ErrorCode::partition_invalid_request,
                  "cell " + std::to_string(c) + " is in part " + std::to_string(parts[c]));
    }
  }

  PartitionQuality q;
  const auto n = at(n_parts);
  q.owned_cells.assign(n, 0);
  q.ghost_cells.assign(n, 0);
  q.part_weights.assign(n, 0);
  for (std::size_t c = 0; c < parts.size(); ++c) {
    ++q.owned_cells[at(parts[c])];
    q.part_weights[at(parts[c])] += cell_weights.empty() ? 1 : cell_weights[c];
  }

  // A ghost of part p is a (p, cell) pair with the cell outside p and
  // adjacent to p; a neighbour of p is a (p, part) pair.
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  std::vector<std::pair<std::int32_t, std::int64_t>> ghosts;
  std::vector<std::pair<std::int32_t, std::int32_t>> links;
  for (std::size_t f = 0; f < neighbour.size(); ++f) {
    const std::int32_t a = parts[at(owner[f])];
    const std::int32_t b = parts[at(neighbour[f])];
    if (a == b) continue;
    ++q.cut_faces;
    ghosts.emplace_back(a, neighbour[f]);
    ghosts.emplace_back(b, owner[f]);
    links.emplace_back(a, b);
    links.emplace_back(b, a);
  }
  std::sort(ghosts.begin(), ghosts.end());
  ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
  for (const auto& [p, cell] : ghosts) ++q.ghost_cells[at(p)];
  std::sort(links.begin(), links.end());
  links.erase(std::unique(links.begin(), links.end()), links.end());
  std::vector<std::int32_t> n_neighbours(n, 0);
  for (const auto& [p, other] : links) ++n_neighbours[at(p)];
  q.max_neighbours = *std::max_element(n_neighbours.begin(), n_neighbours.end());

  std::int64_t total_weight = 0;
  std::int64_t total_ghosts = 0;
  for (std::size_t p = 0; p < n; ++p) {
    total_weight += q.part_weights[p];
    total_ghosts += q.ghost_cells[p];
    if (q.owned_cells[p] > 0) {
      q.max_ghost_ratio = std::max(q.max_ghost_ratio, static_cast<double>(q.ghost_cells[p]) /
                                                          static_cast<double>(q.owned_cells[p]));
    }
  }
  const double mean = static_cast<double>(total_weight) / static_cast<double>(n);
  q.imbalance =
      static_cast<double>(*std::max_element(q.part_weights.begin(), q.part_weights.end())) / mean;
  q.ghost_ratio = static_cast<double>(total_ghosts) / static_cast<double>(mesh.n_cells());
  return q;
}

}  // namespace autan
