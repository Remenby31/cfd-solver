#pragma once

#include <cstdint>
#include <string>

namespace autan::runtime {

/// The number of CUDA devices this process sees: 0 in a build without CUDA,
/// or when the driver reports none or fails.
int visible_gpus();

/// The GPU a rank runs on for devices.kind = cuda: device `local_rank`
/// modulo the visible devices, made current, with a non-blocking stream of
/// its own. Throws std::runtime_error when CUDA fails; the caller has
/// checked that visible_gpus() > 0.
class Gpu {
 public:
  explicit Gpu(int local_rank);
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  ~Gpu();

  [[nodiscard]] int index() const noexcept { return index_; }
  [[nodiscard]] void* stream() const noexcept { return stream_; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  /// The device memory in use now beyond what was free once the stream
  /// existed, by cudaMemGetInfo: on a device the process shares (unified
  /// memory) this counts every process's allocations. Also updates peak().
  std::int64_t sample();
  /// The greatest sample() so far: a lower bound of the true peak, which may
  /// fall between two samples.
  [[nodiscard]] std::int64_t peak() const noexcept { return peak_; }

 private:
  int index_ = -1;
  void* stream_ = nullptr;
  std::string name_;
  std::int64_t free_at_start_ = 0;
  std::int64_t peak_ = 0;
};

}  // namespace autan::runtime
