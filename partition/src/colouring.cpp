#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/partition/partition.hpp"
#include "groups.hpp"

namespace autan {
namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

}  // namespace

std::vector<std::int32_t> colour_greedy(std::span<const offset_t> offsets,
                                        std::span<const index_t> neighbours,
                                        std::span<const std::int64_t> global_id) {
  const std::size_t n = global_id.size();
  if (offsets.size() != n + 1 || offsets.front() != 0 || at(offsets.back()) != neighbours.size() ||
      !std::ranges::is_sorted(offsets)) {
    throw Error(ErrorCode::partition_invalid_request,
                "the graph's offsets do not match its nodes and neighbours");
  }
  for (const index_t v : neighbours) {
    if (v < 0 || at(v) >= n) {
      throw Error(ErrorCode::partition_invalid_request,
                  "neighbour " + std::to_string(v) + " is not a node of the graph");
    }
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, {}, [&](std::size_t v) { return global_id[v]; });
  std::vector<std::int32_t> colour(n, -1);
  std::vector<char> taken;
  for (const std::size_t v : order) {
    taken.assign(taken.size(), 0);
    for (auto k = offsets[v]; k < offsets[v + 1]; ++k) {
      const std::int32_t c = colour[at(neighbours[at(k)])];
      if (c < 0) continue;
      if (at(c) >= taken.size()) taken.resize(at(c) + 1, 0);
      taken[at(c)] = 1;
    }
    std::int32_t c = 0;
    while (at(c) < taken.size() && taken[at(c)] != 0) ++c;
    colour[v] = c;
  }
  return colour;
}

std::vector<std::int32_t> colour_cells(const Mesh& global, std::span<const offset_t> group_offsets,
                                       std::span<const index_t> group_cells) {
  if (global.n_cells() != global.n_owned_cells()) {
    throw Error(ErrorCode::partition_input_has_ghosts,
                "colour_cells colours an unpartitioned mesh");
  }
  // Nodes: cells, or groups and the cells outside them.
  const std::vector<std::int64_t> node_of = detail::group_nodes(global, group_offsets, group_cells);
  const auto ids = global.cell_global_id();
  std::vector<std::int64_t> node_id(ids.begin(), ids.end());
  if (!node_of.empty()) {
    node_id.assign(at(*std::ranges::max_element(node_of) + 1),
                   std::numeric_limits<std::int64_t>::max());
    for (std::size_t c = 0; c < node_of.size(); ++c)
      node_id[at(node_of[c])] = std::min(node_id[at(node_of[c])], ids[c]);
  }
  const auto node = [&](index_t c) {
    return node_of.empty() ? c : static_cast<index_t>(node_of[at(c)]);
  };
  const auto n = node_id.size();
  // Neighbour pairs: across interior faces, and across periodic pairs
  // (the owners of matched faces).
  std::vector<std::pair<index_t, index_t>> pairs;
  for (index_t f = 0; f < global.n_interior_faces(); ++f)
    pairs.emplace_back(node(global.owner()[at(f)]), node(global.neighbour()[at(f)]));
  const auto offsets = global.boundary_zone_offsets();
  const auto periodic = global.periodic_pairs();
  for (std::size_t p = 0; p < periodic.size(); ++p) {
    const auto partners = global.periodic_partners(p);
    const index_t first = offsets[at(periodic[p].zone_a)];
    for (std::size_t k = 0; k < partners.size(); ++k) {
      pairs.emplace_back(node(global.owner()[at(first) + k]),
                         node(global.owner()[at(partners[k])]));
    }
  }
  std::vector<offset_t> csr(n + 1, 0);
  for (const auto& [a, b] : pairs) {
    if (a == b) continue;
    ++csr[at(a) + 1];
    ++csr[at(b) + 1];
  }
  for (std::size_t c = 0; c < n; ++c) csr[c + 1] += csr[c];
  std::vector<index_t> adjacent(at(csr.back()));
  std::vector<offset_t> fill(csr.begin(), csr.end() - 1);
  for (const auto& [a, b] : pairs) {
    if (a == b) continue;
    adjacent[at(fill[at(a)]++)] = b;
    adjacent[at(fill[at(b)]++)] = a;
  }
  const auto colour = colour_greedy(csr, adjacent, node_id);
  if (node_of.empty()) return colour;
  std::vector<std::int32_t> of_cells(node_of.size());
  for (std::size_t c = 0; c < node_of.size(); ++c) of_cells[c] = colour[at(node_of[c])];
  return of_cells;
}

}  // namespace autan
