#include "device.hpp"

#include <stdexcept>

#ifdef AUTAN_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace autan::runtime {

#ifdef AUTAN_HAVE_CUDA

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

int visible_gpus() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

Gpu::Gpu(int local_rank) : index_(local_rank % visible_gpus()) {
  check(cudaSetDevice(index_), "cudaSetDevice");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, index_), "cudaGetDeviceProperties");
  name_ = properties.name;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  stream_ = stream;
  std::size_t free = 0, total = 0;
  check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  free_at_start_ = static_cast<std::int64_t>(free);
}

std::int64_t Gpu::sample() {
  std::size_t free = 0, total = 0;
  check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  const std::int64_t used = free_at_start_ - static_cast<std::int64_t>(free);
  if (used > peak_) peak_ = used;
  return used;
}

Gpu::~Gpu() {
  if (stream_ != nullptr) (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

#else

int visible_gpus() { return 0; }

Gpu::Gpu(int /*local_rank*/) { throw std::runtime_error("this build of autan-run has no CUDA"); }

Gpu::~Gpu() = default;

std::int64_t Gpu::sample() { return 0; }

#endif

}  // namespace autan::runtime
