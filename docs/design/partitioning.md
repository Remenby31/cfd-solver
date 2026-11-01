# Partitioning and halo exchange

How a mesh is cut into one partition per rank and how ranks exchange cell values. Code: `partition/`, libraries `autan::partition` (no MPI) and `autan::halo` (MPI). The partition layout itself is part of the [mesh contract](../contracts/mesh.md).

## Partitioning

`partition_cells(mesh, n_parts, options)` cuts the dual graph of an unpartitioned mesh: one vertex per cell, one edge per pair of cells that share at least one interior face.

| Method | Library | Licence | Build option |
|---|---|---|---|
| `metis` (default) | METIS 5 `METIS_PartGraphKway` | Apache-2.0 | `AUTAN_ENABLE_METIS` |
| `scotch` | Scotch 7 `SCOTCH_graphPart`, quality strategy | CeCILL-C | `AUTAN_ENABLE_SCOTCH` |

ParMETIS is excluded by its licence. Both methods are seeded (`options.seed`), and Scotch runs on one thread in a private, seeded context, so a mesh and its options give the same parts on any machine and any core count. Scotch's deterministic option alone is not enough: with 8 or more threads its k-way partitioner returned different parts for the same call. A method the build lacks throws E0142; nothing falls back to the other.

`options.cell_weights` balances cost instead of cell count. `cell_cost_weights(mesh)` gives the number of faces of each cell, the cost of a face loop: a 100-face polyhedron weighs as much as about 17 hexahedra.

Partitioning is serial: every rank reads the global mesh, rank 0 partitions it and broadcasts the parts, and each rank extracts its own share. Distributed reading and PT-Scotch's parallel partitioner come with distributed readers.

## A partition

`extract_partition(global, compute_geometry(global), parts, n_parts, rank)` returns a `LocalPartition`:

- `mesh`: owned cells in the order of the global mesh, then the ghost cells within `layers` face steps (2 by default, 1 on request), ordered by owning rank, the same order, then periodic shift. The first ghost layer carries all its faces, so the face-of-face stencil of every owned cell is local; those halo faces come after the owned faces (mesh contract, invariant 6), and loops that write residuals stop at `n_owned_interior_faces` and `n_owned_faces`. Each cell keeps its `cell_global_id`, so a global mesh renumbered before partitioning keeps its ids. Built through `assemble`, so it meets every invariant of the mesh contract.
- `geometry`: the global geometry, copied bit for bit. A face whose orientation flips (its owned cell has the higher global index) has its area vector negated, which is exact. Ghost cells are NaN until a forward exchange fills them (`exchange_ghost_geometry`). `compute_geometry` on the partition would agree only to round-off, because a reversed face sums its nodes in another order.
- `face_global_id`: the global face of each local face, the key for summing a cell's faces in an order that no partition changes. For a periodic face it is the face that represents the pair, the same on every part.
- `face_reversed`: 1 where the local face runs opposite to its global face (the negated area vectors above), so that a kernel can evaluate a flux in the global orientation, swapping its two states, and get the global bits; never on a boundary face.
- `halo`: the `HaloPattern` below.

## Halo pattern

Ghosts from neighbour `i` are the contiguous cells `n_owned + [recv_offsets[i], recv_offsets[i+1])`. The owned cells that rank `i` holds as ghosts are `send_cells[send_offsets[i] .. send_offsets[i+1])`, in the same order: a cell is a ghost of part `q` exactly when it is within `layers` face steps of `q`, a symmetric relation, so both lists are the same cells in the same order. `interface_faces` lists the owned interior faces that touch a ghost.

## Periodic images

A mesh with periodic pairs is cut in its periodic covering: a cell copy is a global cell and a shift, one integer per direction, at the cell's position plus the shifts times the directions. Pairs whose translations are equal or opposite share a direction; at most three directions, linearly independent, else E0140, so that a shift names one translation. Across a periodic face, a cell's neighbour is the partner face's cell, shifted by one translation. Each physical periodic face appears once in a partition, as an interior face drawn from its zone-a face; the nodes of both sides are merged through the matched node pairs (one class per physical point, with a shift per node), so cells stay closed. A ghost that is a shifted copy is an image: `ghost_translation` holds its shift, its nodes and faces are translated, and `exchange_ghost_geometry` adds the translation to its centroid after the exchange. Cell values need no change under a translation. Images of a part's own cells make the part its own neighbour: on one rank too, the exchange then moves them by a message to self. Without MPI, `forward_self` copies them on the host and `DeviceSelfForward` in device memory, on a stream (E0145 in a build without CUDA).

## Exchange

`HaloExchange(pattern, n_owned, comm, space)` checks with one all-to-all that every send count matches the receiver's ghost count (E0144 otherwise), allocates its buffers once in `space`, and moves 1 to 16 components of `float` or `double` per call.

| Operation | Effect | Use |
|---|---|---|
| `forward` | ghost ← owner value | residuals, gradients, geometry |
| `reverse` | owner += Σ ghost values held by other ranks; ghosts ← 0 | partial sums of a face loop scattered to ghosts, adjoint (transposed) kernels |

`reverse` is the transpose of `forward`: for any `u` and `v` over all owned and ghost entries of all ranks, ⟨forward(u), v⟩ = ⟨u, reverse(v)⟩.

With `MemorySpace::cuda`, packing and unpacking run as CUDA kernels and device buffers go straight to a CUDA-aware MPI (Open MPI over UCX in the dev image). A build without CUDA, or an MPI that `MPIX_Query_cuda_support` reports as not CUDA-aware, throws E0145: there is no host staging path.

**Overlap.** `begin_forward(field, stream)` posts the receives, packs on `stream`, waits for the pack and posts the sends; `end_forward()` waits for the messages and unpacks. Between the two, a kernel may process every face except `interface_faces`. `begin_reverse` / `end_reverse` work the same way; between them the ghosts must not be written.

**Determinism.** No exchange uses atomics. Reverse sums each owned cell's contributions in FP64 in ascending neighbour rank, then rounds once, so a run repeats bit for bit. A gather kernel that reads ghosts after a forward exchange and sums each cell's faces in ascending `face_global_id` gives the same bits on any number of ranks, since geometry and ghost values are identical to the global ones. A reverse sum groups contributions by rank, so its result across different rank counts agrees to round-off, not bit for bit.

## Colouring

`colour_greedy(offsets, neighbours, global_id)` colours a graph (CSR) greedily, visiting its nodes in increasing global id: each takes the least colour none of its already coloured neighbours has. The colours depend on the graph and the ids only, not on the order of the nodes, so every rank of a run, and a renumbered mesh (RCM), compute the same ones. `colour_cells(mesh)` applies it to the cells of an unpartitioned mesh, neighbours across interior faces and across periodic pairs, ids `cell_global_id`: two cells a smoother updates from each other never share a colour. A multicolour smoother that sweeps these colours and exchanges ghosts after each colour therefore does on N ranks what it does on one (#259). A line smoother passes its lines as groups: `colour_cells(mesh, group_offsets, group_cells)` makes each group one node, its id the least `cell_global_id` of its cells, and gives every cell its group's colour.

## Groups and graph levels

`PartitionOptions::group_offsets` and `group_cells` name groups of cells that must share a part, a smoother's lines for instance. `partition_cells` contracts each group into one node of the graph its method partitions, weighing the sum of its cells, then gives every cell its group's part.

A multigrid level is a graph, `GraphLevel`: nodes with global ids, and edges between them. `coarse_parts` gives each coarse node the part of its child with the least global id; every rank holds the same hierarchy, so all agree. `extract_graph` builds a rank's share of a level, `LocalGraph`:
- **Nodes:** owned nodes in increasing global id, then ghost nodes ordered by owning part and global id. The ghosts are the edge neighbours of the owned nodes and, with coarse parts given, the remote children (`remote_children`: children, owned elsewhere, of the coarse nodes this rank owns) and their neighbours.
- **Edges:** those with an end owned or a remote child, each owned by the part of its owner node, in global orientation flagged by `edge_reversed`.
- **Halos:** `node_halo` and `edge_halo` are HaloPatterns, so `HaloExchange` moves per-node and per-edge data. A coarse owner then restricts and assembles from all its children's values, summed in global id order: the same bits on any number of ranks, with no reverse sum (#259).

## Groups and graph levels

`PartitionOptions::group_offsets` and `group_cells` name groups of cells that must share a part, a smoother's lines for instance. `partition_cells` contracts each group into one node of the graph its method partitions, weighing the sum of its cells, then gives every cell its group's part.

A multigrid level is a graph, `GraphLevel`: nodes with global ids, and edges between them. `coarse_parts` gives each coarse node the part of its child with the least global id; every rank holds the same hierarchy, so all agree. `extract_graph` builds a rank's share of a level, `LocalGraph`:
- **Nodes:** owned nodes in increasing global id, then ghost nodes ordered by owning part and global id. The ghosts are the edge neighbours of the owned nodes and, with coarse parts given, the remote children (`remote_children`: children, owned elsewhere, of the coarse nodes this rank owns) and their neighbours.
- **Edges:** those with an end owned or a remote child, each owned by the part of its owner node, in global orientation flagged by `edge_reversed`.
- **Halos:** `node_halo` and `edge_halo` are HaloPatterns, so `HaloExchange` moves per-node and per-edge data. A coarse owner then restricts and assembles from all its children's values, summed in global id order: the same bits on any number of ranks, with no reverse sum (#259).

## Tests

`tests/partition/`: analytic halo sizes of slab partitions, valid extraction for every cell type and a 105-face polyhedron, balance and reproducibility of METIS and Scotch, and error codes. `autan_halo_tests` runs on 1, 2, 4 and 8 ranks, on the host and on the GPU: ghost geometry and a Green–Gauss gradient match the single-rank values bit for bit in FP32 and FP64, reverse passes the dot-product test against forward, and reverse counts the ranks that hold each cell as a ghost.
