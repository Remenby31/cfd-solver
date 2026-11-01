#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan::detail {

// Each cell's node once the groups (CSR) contract: groups first in their
// order, then the other cells in theirs; empty when there is no group.
// Throws E0140 for malformed offsets, an empty group, or a cell out of range
// or in two groups.
std::vector<std::int64_t> group_nodes(const Mesh& mesh, std::span<const offset_t> offsets,
                                      std::span<const index_t> group_cells);

}  // namespace autan::detail
