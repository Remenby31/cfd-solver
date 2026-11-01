#pragma once

// Device copies of a spike mesh and its geometry, shared by the GPU driver and the benchmark.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/kernels/gpu.hpp"
#include "spike.hpp"

namespace autan::spike {

using autan::kernels::index_t;
using autan::kernels::offset_t;
using autan::spike::GeomArray;
namespace gpu = autan::kernels::gpu;

/// Cells per slice of the stencil layout of every DeviceMesh built afterwards (0: CSR).
inline int& stencil_slice() {
  static int slice = 0;
  return slice;
}

/// Cells per slice of the incidence layout of every DeviceMesh built afterwards (0: CSR).
inline int& incidence_slice() {
  static int slice = 0;
  return slice;
}

template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    gpu::check(cudaMalloc(&p_, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    gpu::check(cudaMemset(p_, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
  }
  explicit DeviceArray(const std::vector<T>& v) : DeviceArray(v.size()) { upload(v.data()); }
  DeviceArray(const T* host, std::size_t n) : DeviceArray(n) { upload(host); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(p_); }
  void upload(const T* host) {
    gpu::check(cudaMemcpy(p_, host, n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<T> download() const {
    std::vector<T> v(n_);
    gpu::check(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
  }
  T* get() const { return p_; }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

/// A spike mesh on the device through the library's DeviceMesh; `topo` and `schedule` are
/// copies for the call sites that set them directly.
struct DeviceMesh {
  static constexpr int kScratchWidth = 10;          // widest spike face loop: the Roe adjoint
  static constexpr int kIncidenceScratchWidth = 1;  // spike reductions gather one scalar
  gpu::DeviceMesh mesh;
  autan::kernels::Topology topo;
  gpu::Schedule schedule;

  DeviceMesh(const SpikeMesh& m, int block) : DeviceMesh(m, block, m.keys()) {}
  DeviceMesh(const SpikeMesh& m, int block, const autan::kernels::GlobalKeys& keys)
      : mesh(m.mesh, m.packed, m.topo.stencil != nullptr ? &m.stencil : nullptr,
             {block, kScratchWidth, kIncidenceScratchWidth, nullptr, stencil_slice(),
              m.global ? &keys : nullptr, incidence_slice()}),
        topo(mesh.topology()),
        schedule(mesh.schedule()) {}
  template <class D>
  void bind(D& d) {
    mesh.bind(d);
  }
};

using autan::kernels::strategy_of;

}  // namespace autan::spike
