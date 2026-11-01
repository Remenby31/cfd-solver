#pragma once

// Device copies of what generated loops read besides a physics' own datasets, and exact device
// reductions. CUDA and HIP translation units only.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include "autan/core/mesh.hpp"
#include "autan/kernels/colouring.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/gpu.hpp"
#include "autan/kernels/runtime.hpp"

namespace autan::kernels::gpu {

namespace detail {
#if defined(AUTAN_GPU_HIP)
inline error_t allocate(void** p, std::size_t bytes) { return hipMalloc(p, bytes); }
inline void release(void* p) { static_cast<void>(hipFree(p)); }
inline error_t zero(void* p, std::size_t bytes, stream_t s) {
  return hipMemsetAsync(p, 0, bytes, s);
}
inline error_t to_device(void* d, const void* h, std::size_t bytes, stream_t s) {
  return hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s);
}
inline error_t to_host(void* h, const void* d, std::size_t bytes, stream_t s) {
  return hipMemcpyAsync(h, d, bytes, hipMemcpyDeviceToHost, s);
}
inline error_t synchronize(stream_t s) { return hipStreamSynchronize(s); }
#else
inline error_t allocate(void** p, std::size_t bytes) { return cudaMalloc(p, bytes); }
inline void release(void* p) { static_cast<void>(cudaFree(p)); }
inline error_t zero(void* p, std::size_t bytes, stream_t s) {
  return cudaMemsetAsync(p, 0, bytes, s);
}
inline error_t to_device(void* d, const void* h, std::size_t bytes, stream_t s) {
  return cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);
}
inline error_t to_host(void* h, const void* d, std::size_t bytes, stream_t s) {
  return cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s);
}
inline error_t synchronize(stream_t s) { return cudaStreamSynchronize(s); }
#endif
}  // namespace detail

/// Owning device array of n values, zero-filled. Copies run on `stream` and return once done,
/// so host buffers may be reused at once.
template <class T>
class DeviceArray {
  static_assert(std::is_trivially_copyable_v<T>);

 public:
  DeviceArray() = default;
  explicit DeviceArray(std::size_t n, stream_t stream = nullptr) : n_(n) {
    check(detail::allocate(reinterpret_cast<void**>(&p_), std::max<std::size_t>(n, 1) * sizeof(T)),
          "device allocation");
    check(detail::zero(p_, std::max<std::size_t>(n, 1) * sizeof(T), stream), "device zero");
    check(detail::synchronize(stream), "device zero");
  }
  explicit DeviceArray(std::span<const T> host, stream_t stream = nullptr)
      : DeviceArray(host.size(), stream) {
    upload(host, stream);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(o.n_) {}
  DeviceArray& operator=(DeviceArray&& o) noexcept {
    std::swap(p_, o.p_);
    std::swap(n_, o.n_);
    return *this;
  }
  ~DeviceArray() {
    if (p_ != nullptr) {
      detail::release(p_);
    }
  }
  void upload(std::span<const T> host, stream_t stream = nullptr) {
    if (host.size() != n_) {
      throw std::invalid_argument("E1101: upload of " + std::to_string(host.size()) +
                                  " values into a device array of " + std::to_string(n_));
    }
    check(detail::to_device(p_, host.data(), n_ * sizeof(T), stream), "upload");
    check(detail::synchronize(stream), "upload");
  }
  void download(std::span<T> host, stream_t stream = nullptr) const {
    if (host.size() != n_) {
      throw std::invalid_argument("E1101: download of a device array of " + std::to_string(n_) +
                                  " values into " + std::to_string(host.size()));
    }
    check(detail::to_host(host.data(), p_, n_ * sizeof(T), stream), "download");
    check(detail::synchronize(stream), "download");
  }
  [[nodiscard]] std::vector<T> download(stream_t stream = nullptr) const {
    std::vector<T> v(n_);
    download(std::span<T>(v), stream);
    return v;
  }
  [[nodiscard]] T* get() const { return p_; }
  [[nodiscard]] std::size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

/// What the generated loops of any module read besides the physics' own datasets: topology
/// (owned-face prefixes, incidence twins, cell stencil and its twins), built-in geometry,
/// colourings of the owned face ranges, store-gather and incidence scratch, and the Schedule.
/// Not tied to a policy: geometry is uploaded in the type each generated Data member stores.
class DeviceMesh {
 public:
  struct Options {
    int block = 256;
    int face_scratch_width = 0;       ///< components per face of the widest store-gather loop
    int incidence_scratch_width = 0;  ///< per entry, of the widest deterministic reduction adjoint
    stream_t stream = nullptr;
    /// Cells per slice of the sliced stencil layout (Topology::stencil_stride), 0 for CSR. Same
    /// results bit for bit; a warp then reads its cells' stencil entries contiguously.
    int stencil_slice = 0;
    /// The partition's global numbering (incidences by global face id, reversed faces evaluated
    /// in their global orientation), for results that do not depend on the partition; the
    /// geometry and stencil passed must then be built with the same keys. Null: the mesh's order.
    const GlobalKeys* global = nullptr;
    /// Cells per slice of the sliced incidence layout (Topology::incidence_stride, with the
    /// column index incidence_cell), 0 for CSR. Same results bit for bit. 32 for the block
    /// solvers (kernels/blocks.py: the product reaches 88-93 % of GB10 peak, CSR 31-68 %); CSR
    /// by default, the explicit Euler step being no faster sliced (+0.4 % tet, +1.1 % hex).
    int incidence_slice = 0;
  };

  /// `stencil` may be null (no stencil reductions); it is copied.
  DeviceMesh(const Mesh& mesh, const PackedGeometry& geometry, const CellStencil* stencil,
             Options options)
      : geometry_(geometry), stream_(options.stream) {
    Topology host = kernels::topology(mesh.view());
    GlobalOrder order;
    if (options.global != nullptr) {
      order = global_order(mesh, *options.global);
      host.cell_faces = order.cell_faces.data();
    } else {
      order.twin = incidence_twins(mesh);
    }
    auto& twins = order.twin;
    host.incidence_twin = twins.data();
    if (options.incidence_slice > 0) {
      incidences_ = sliced_incidences(host, options.incidence_slice);
      geometry_.lsq_weight = sliced_values(incidences_, geometry_.lsq_weight, 3);
      geometry_.incidence_side = sliced_values(incidences_, geometry_.incidence_side, 1);
      twins = incidences_.twin;
      host.incidence_twin = twins.data();
      host.n_incidences = static_cast<offset_t>(incidences_.cell_faces.size());
      host.cell_face_offsets = incidences_.cell_face_offsets.data();
      host.cell_faces = incidences_.cell_faces.data();
      host.incidence_stride = incidences_.stride;
      host.incidence_length = incidences_.length.data();
      host.incidence_cell = incidences_.cell.data();
    }
    if (stencil != nullptr) {
      csr_entries_ = stencil->entries.size();
      if (options.stencil_slice > 0) {
        auto s = sliced(*stencil, options.stencil_slice);
        stencil_ = std::move(s.stencil);
        position_ = std::move(s.position);
        length_ = std::move(s.length);
        host.stencil_stride = s.stride;
        host.stencil_length = length_.data();
      } else {
        stencil_ = *stencil;
      }
      geometry_.stencil_lsq_weight = stencil_.lsq_weight;
      host.n_stencil = static_cast<offset_t>(stencil_.entries.size());
      host.stencil_offsets = stencil_.offsets.data();
      host.stencil = stencil_.entries.data();
      host.stencil_twin = stencil_.twin.data();
    }
    const auto nf = static_cast<std::size_t>(host.n_faces);
    owner_ = copy(std::span(host.owner, nf));
    neighbour_ = copy(std::span(host.neighbour, static_cast<std::size_t>(host.n_interior_faces)));
    offsets_ = copy(std::span(host.cell_face_offsets, static_cast<std::size_t>(host.n_cells) + 1));
    cell_faces_ = copy(std::span(host.cell_faces, static_cast<std::size_t>(host.n_incidences)));
    twin_ = copy(std::span<const offset_t>(twins));
    interior_ = colour_faces(host, 0, host.n_owned_interior_faces);
    boundary_ = colour_faces(host, host.n_interior_faces, host.n_owned_faces);
    interior_faces_ = copy(std::span<const index_t>(interior_.faces));
    boundary_faces_ = copy(std::span<const index_t>(boundary_.faces));
    topo_ = host;
    topo_.owner = owner_.get();
    topo_.neighbour = neighbour_.get();
    topo_.cell_face_offsets = offsets_.get();
    topo_.cell_faces = cell_faces_.get();
    topo_.incidence_twin = twin_.get();
    if (options.global != nullptr && !options.global->reversed.empty()) {
      reversed_ = copy(options.global->reversed);
      topo_.face_reversed = reversed_.get();
    }
    topo_.rank_independent = options.global != nullptr;
    if (options.incidence_slice > 0) {
      incidence_length_ = copy(std::span<const index_t>(incidences_.length));
      incidence_cell_ = copy(std::span<const index_t>(incidences_.cell));
      topo_.incidence_length = incidence_length_.get();
      topo_.incidence_cell = incidence_cell_.get();
    }
    if (stencil != nullptr) {
      stencil_offsets_ = copy(std::span<const offset_t>(stencil_.offsets));
      stencil_entries_ = copy(std::span<const index_t>(stencil_.entries));
      stencil_twin_ = copy(std::span<const offset_t>(stencil_.twin));
      topo_.stencil_offsets = stencil_offsets_.get();
      topo_.stencil = stencil_entries_.get();
      topo_.stencil_twin = stencil_twin_.get();
      if (!length_.empty()) {
        stencil_length_ = copy(std::span<const index_t>(length_));
        topo_.stencil_length = stencil_length_.get();
      }
    }
    // Scratch holds compute values (face) or accumulation values (incidence): at most 8 bytes.
    face_scratch_ = DeviceArray<double>(
        static_cast<std::size_t>(std::max(options.face_scratch_width, 0)) * nf, stream_);
    incidence_scratch_ = DeviceArray<double>(
        static_cast<std::size_t>(std::max(options.incidence_scratch_width, 0)) *
            static_cast<std::size_t>(std::max(host.n_incidences, host.n_stencil)),
        stream_);
    schedule_.block = options.block;
    schedule_.interior = {interior_faces_.get(), interior_.offsets};
    schedule_.boundary = {boundary_faces_.get(), boundary_.offsets};
    schedule_.face_scratch = face_scratch_.get();
    schedule_.face_scratch_width = options.face_scratch_width;
    schedule_.incidence_scratch = incidence_scratch_.get();
    schedule_.incidence_scratch_width = options.incidence_scratch_width;
    schedule_.stream = stream_;
  }
  DeviceMesh(const DeviceMesh&) = delete;
  DeviceMesh& operator=(const DeviceMesh&) = delete;

  [[nodiscard]] const Topology& topology() const { return topo_; }
  [[nodiscard]] const Schedule& schedule() const { return schedule_; }

  /// Sets d.topo and points every built-in geometry member of a generated Data struct at a
  /// device copy in that member's own type (P::geom or P::geom_value), uploaded on first use.
  /// Built-in geometry is read-only (`const T*`); a physics' own dataset of the same name is
  /// writable and left alone.
  template <class Data>
  void bind(Data& d) {
    d.topo = topo_;
#define AUTAN_BIND_GEOMETRY(name)                                                    \
  if constexpr (requires { d.name; }) {                                              \
    if constexpr (std::is_const_v<std::remove_pointer_t<decltype(d.name)>>) {        \
      d.name = field<std::remove_cvref_t<decltype(*d.name)>>(#name, geometry_.name); \
    }                                                                                \
  }
    AUTAN_BIND_GEOMETRY(face_area)
    AUTAN_BIND_GEOMETRY(face_centroid)
    AUTAN_BIND_GEOMETRY(face_normal)
    AUTAN_BIND_GEOMETRY(face_area_magnitude)
    AUTAN_BIND_GEOMETRY(face_r_left)
    AUTAN_BIND_GEOMETRY(face_r_right)
    AUTAN_BIND_GEOMETRY(face_moment)
    AUTAN_BIND_GEOMETRY(cell_centroid)
    AUTAN_BIND_GEOMETRY(cell_volume)
    AUTAN_BIND_GEOMETRY(lsq_weight)
    AUTAN_BIND_GEOMETRY(incidence_side)
    AUTAN_BIND_GEOMETRY(stencil_lsq_weight)
#undef AUTAN_BIND_GEOMETRY
  }

  /// Replaces what bind() gives stencil_lsq_weight: 3 x n values in the entry order of the
  /// stencil given at construction (component-major, n its entries), whatever the device
  /// layout. Data structs bound before keep the previous weights, still allocated, until bound
  /// again.
  void set_stencil_weights(std::span<const double> weights) {
    if (weights.size() != 3 * csr_entries_) {
      throw std::invalid_argument("E1101: stencil weights of the wrong size");
    }
    if (position_.empty()) {
      geometry_.stencil_lsq_weight.assign(weights.begin(), weights.end());
    } else {
      const std::size_t n = stencil_.entries.size();
      geometry_.stencil_lsq_weight.assign(3 * n, 0.0);
      for (std::size_t k = 0; k < csr_entries_; ++k) {
        for (std::size_t i = 0; i < 3; ++i) {
          geometry_.stencil_lsq_weight[i * n + static_cast<std::size_t>(position_[k])] =
              weights[i * csr_entries_ + k];
        }
      }
    }
    for (auto it = fields_.begin(); it != fields_.end();) {
      if (it->first.first == "stencil_lsq_weight") {
        retired_.push_back(std::move(it->second));
        it = fields_.erase(it);
      } else {
        ++it;
      }
    }
  }

 private:
  template <class T>
  DeviceArray<T> copy(std::span<const T> host) {
    return DeviceArray<T>(host, stream_);
  }

  template <class T>
  const T* field(const std::string& name, const std::vector<double>& host) {
    const auto key = std::make_pair(name, std::type_index(typeid(T)));
    auto it = fields_.find(key);
    if (it == fields_.end()) {
      const auto values = converted<T>(host);
      DeviceArray<std::byte> a(values.size() * sizeof(T), stream_);
      a.upload(std::as_bytes(std::span<const T>(values)), stream_);
      it = fields_.emplace(key, std::move(a)).first;
    }
    return reinterpret_cast<const T*>(it->second.get());
  }

  PackedGeometry geometry_;
  CellStencil stencil_;
  std::size_t csr_entries_ = 0;
  std::vector<offset_t> position_;  // sliced layout: where each CSR entry went
  std::vector<index_t> length_;
  stream_t stream_;
  DeviceArray<index_t> owner_, neighbour_, cell_faces_, interior_faces_, boundary_faces_;
  DeviceArray<offset_t> offsets_, twin_;
  DeviceArray<std::uint8_t> reversed_;
  DeviceArray<offset_t> stencil_offsets_, stencil_twin_;
  DeviceArray<index_t> stencil_entries_, stencil_length_;
  SlicedIncidences incidences_;
  DeviceArray<index_t> incidence_length_, incidence_cell_;
  HostColouring interior_, boundary_;
  DeviceArray<double> face_scratch_, incidence_scratch_;
  std::map<std::pair<std::string, std::type_index>, DeviceArray<std::byte>> fields_;
  std::vector<DeviceArray<std::byte>> retired_;  // replaced weights some Data may still point at
  Topology topo_{};
  Schedule schedule_;
};

namespace detail {

// Ties between -0 and +0 go to -0, so the result does not depend on the order of comparisons.
template <class T>
__device__ inline bool before(T a, T b) {
  return a < b || (a == b && std::signbit(a) && !std::signbit(b));
}

template <class T>
__global__ void min_blocks(index_t n, const T* __restrict__ v, T* __restrict__ out) {
  __shared__ T s[256];
  const auto t = static_cast<int>(threadIdx.x);
  T m = infinity<T>();
  for (auto i = static_cast<offset_t>(blockIdx.x) * 256 + t; i < n;
       i += static_cast<offset_t>(gridDim.x) * 256) {
    if (before(v[i], m)) {
      m = v[i];
    }
  }
  s[t] = m;
  __syncthreads();
  for (int h = 128; h > 0; h /= 2) {
    if (t < h && before(s[t + h], s[t])) {
      s[t] = s[t + h];
    }
    __syncthreads();
  }
  if (t == 0) {
    out[blockIdx.x] = s[0];
  }
}

template <class F>
__global__ void first_flag(index_t n, const F* __restrict__ flags, index_t* __restrict__ out) {
  const auto i = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n && flags[i] != F(0)) {
    atomicMin(out, static_cast<index_t>(i));
  }
}

}  // namespace detail

/// Smallest of values[0..n) (NaN never selected; -0 before +0); +infinity when n == 0. The
/// result is exact, so it does not depend on the launch.
template <class T>
T min(index_t n, const T* values, stream_t stream = nullptr) {
  constexpr int blocks = 256;
  DeviceArray<T> partial(blocks, stream);
  if (n > 0) {
    detail::min_blocks<T><<<blocks, 256, 0, stream>>>(n, values, partial.get());
    detail::min_blocks<T><<<1, 256, 0, stream>>>(blocks, partial.get(), partial.get());
    check(last_error(), "gpu::min");
  }
  std::vector<T> h(blocks);
  partial.download(std::span<T>(h), stream);
  return n > 0 ? h[0] : infinity<T>();
}

/// Smallest i < n with flags[i] != 0, or n if none; flags of any arithmetic type (a generated
/// cell dataset written 0 or 1, for instance).
template <class F>
index_t first_index(index_t n, const F* flags, stream_t stream = nullptr) {
  DeviceArray<index_t> out(1, stream);
  out.upload(std::span<const index_t>(&n, 1), stream);
  if (n > 0) {
    detail::first_flag<F>
        <<<static_cast<unsigned>((offset_t(n) + 255) / 256), 256, 0, stream>>>(n, flags, out.get());
    check(last_error(), "gpu::first_index");
  }
  index_t r = n;
  out.download(std::span<index_t>(&r, 1), stream);
  return r;
}

}  // namespace autan::kernels::gpu
