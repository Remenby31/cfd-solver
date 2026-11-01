#pragma once

// 5 x 5 row-major blocks shared by the smoothers of solvers/.
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace autan::solvers::detail {

inline constexpr std::size_t b2 = 25;

// Inverse of a 5 x 5 row-major block by Gauss-Jordan elimination with partial pivoting.
inline void invert5(const double* a, double* inv) {
  std::array<double, 50> m{};
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      m[10 * i + j] = a[5 * i + j];
      m[10 * i + 5 + j] = i == j ? 1.0 : 0.0;
    }
  }
  for (std::size_t k = 0; k < 5; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < 5; ++i) {
      if (std::abs(m[10 * i + k]) > std::abs(m[10 * p + k])) p = i;
    }
    if (m[10 * p + k] == 0.0 || !std::isfinite(m[10 * p + k])) {
      throw std::domain_error("singular diagonal block");
    }
    if (p != k) {
      for (std::size_t j = 0; j < 10; ++j) std::swap(m[10 * k + j], m[10 * p + j]);
    }
    const double d = 1.0 / m[10 * k + k];
    for (std::size_t j = 0; j < 10; ++j) m[10 * k + j] *= d;
    for (std::size_t i = 0; i < 5; ++i) {
      if (i == k) continue;
      const double f = m[10 * i + k];
      for (std::size_t j = 0; j < 10; ++j) m[10 * i + j] -= f * m[10 * k + j];
    }
  }
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 5; ++j) inv[5 * i + j] = m[10 * i + 5 + j];
  }
}

// c = a b, 5 x 5 row-major.
inline void multiply5(const double* a, const double* b, double* c) {
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < 5; ++k) s += a[5 * i + k] * b[5 * k + j];
      c[5 * i + j] = s;
    }
  }
}

inline void add5(double* a, const double* b) {
  for (std::size_t k = 0; k < b2; ++k) a[k] += b[k];
}

}  // namespace autan::solvers::detail
