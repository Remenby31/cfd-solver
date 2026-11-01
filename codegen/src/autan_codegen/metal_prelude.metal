// Prelude of every generated Metal library: the Metal spelling of autan/kernels/runtime.hpp,
// the double-float accumulation type and the argument blocks of the generated kernels. Metal
// has no FP64, so the fp32+acc64 policy keeps its FP64 roles (residual, accumulation, geometry,
// parameters) as unevaluated sums of two floats.
#include <metal_stdlib>

namespace autan {
namespace kernels {

using metal::cos;
using metal::exp;
using metal::fma;
using metal::log;
using metal::sin;
using metal::sqrt;

typedef int index_t;
typedef long offset_t;

// Same layout as autan::kernels::Topology on the host, which fills it with GPU addresses.
struct Topology {
  index_t n_cells;
  index_t n_owned_cells;
  index_t n_faces;
  index_t n_interior_faces;
  index_t n_owned_interior_faces;
  index_t n_owned_faces;
  offset_t n_incidences;
  device const index_t* owner;
  device const index_t* neighbour;
  device const offset_t* cell_face_offsets;
  device const index_t* cell_faces;
  device const offset_t* incidence_twin;
  offset_t n_stencil;
  device const offset_t* stencil_offsets;
  device const index_t* stencil;
  device const offset_t* stencil_twin;
  offset_t stencil_stride;
  device const index_t* stencil_length;
  device const uchar* face_reversed;
  bool rank_independent;
  offset_t incidence_stride;
  device const index_t* incidence_length;
  device const index_t* incidence_cell;
};

// Double-float: x = hi + lo with |lo| <= ulp(hi) / 2, about 48 significant bits.
// TwoSum: Knuth, TAOCP vol. 2, 4.2.2, theorem B; Fast2Sum: Dekker (1971), needs |a| >= |b|.
// Sum: Joldes, Muller and Popescu (2017), "Tight and rigorous error bounds for basic building
// blocks of double-word arithmetic", algorithm 6 (AccurateDWPlusDW), relative error <= 3u^2;
// algorithm 4 (DWPlusFP), relative error <= 2u^2, when y is a float (y.lo == 0): accumulations
// of body outputs, half the operations.
// Product: Dekker (1971), the exact error of hi * hi from an FMA, lo * lo dropped.
// Exact only without reassociation: the library is compiled with MTLMathModeSafe.
struct df64 {
  float hi;
  float lo;
  df64() = default;
  df64(float x) : hi(x), lo(0.0f) {}
  df64(int x) : hi(float(x)), lo(0.0f) {}
  df64(float h, float l) : hi(h), lo(l) {}
  explicit operator float() const { return hi + lo; }
};

inline df64 two_sum(float a, float b) {
  const float s = a + b;
  const float bb = s - a;
  return df64(s, (a - (s - bb)) + (b - bb));
}

inline df64 fast_two_sum(float a, float b) {
  const float s = a + b;
  return df64(s, b - (s - a));
}

inline df64 operator+(df64 x, df64 y) {
  if (y.lo == 0.0f) {
    const df64 s = two_sum(x.hi, y.hi);
    return fast_two_sum(s.hi, x.lo + s.lo);
  }
  const df64 s = two_sum(x.hi, y.hi);
  const df64 t = two_sum(x.lo, y.lo);
  const df64 v = fast_two_sum(s.hi, s.lo + t.hi);
  return fast_two_sum(v.hi, t.lo + v.lo);
}

inline df64 operator-(df64 x) { return df64(-x.hi, -x.lo); }
inline df64 operator-(df64 x, df64 y) { return x + (-y); }

inline df64 operator*(df64 x, df64 y) {
  const float p = x.hi * y.hi;
  const float e = fma(x.hi, y.hi, -p);
  return fast_two_sum(p, fma(x.hi, y.lo, fma(x.lo, y.hi, e)));
}

inline thread df64& operator+=(thread df64& a, df64 b) {
  a = a + b;
  return a;
}
inline device df64& operator+=(device df64& a, df64 b) {
  a = a + b;
  return a;
}
inline device df64& operator-=(device df64& a, df64 b) {
  a = a - b;
  return a;
}

inline bool operator<(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool operator>(df64 a, df64 b) { return b < a; }
inline bool operator<=(df64 a, df64 b) { return !(b < a); }
inline bool operator>=(df64 a, df64 b) { return !(a < b); }

template <class T>
inline bool lt(T a, T b) {
  return a < b;
}
template <class T>
inline bool le(T a, T b) {
  return a <= b;
}
template <class T>
inline bool gt(T a, T b) {
  return a > b;
}
template <class T>
inline bool ge(T a, T b) {
  return a >= b;
}

template <class T>
inline T infinity();
template <>
inline float infinity<float>() {
  return INFINITY;
}
template <>
inline df64 infinity<df64>() {
  return df64(INFINITY, 0.0f);
}

// Device-scope relaxed float atomics (MSL 3.0, section 6.15.2). There is no 64-bit
// floating-point atomic, so the double-float policy has none.
inline void atomicAdd(device float* p, float v) {
  metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic_float*>(p), v,
                                   metal::memory_order_relaxed);
}

// Precision policies: autan::kernels::metal::fp32 and fp32_acc64 on the host.
struct fp32 {
  typedef float state;
  typedef float grad;
  typedef float residual;
  typedef float geom;
  typedef float geom_value;
  typedef float jac;
  typedef float param;
  typedef float compute;
  typedef float accum;
};

struct fp32_acc64 {
  typedef float state;
  typedef float grad;
  typedef df64 residual;
  typedef df64 geom;
  typedef float geom_value;
  typedef float jac;
  typedef df64 param;
  typedef float compute;
  typedef df64 accum;
};

// Argument blocks of the generated kernels (buffer 1); host mirrors in autan/kernels/metal.hpp.
struct RangeArgs {
  index_t lo;
  index_t hi;
};

struct ColourArgs {
  device const index_t* faces;
  index_t count;
};

template <class A>
struct StoreArgs {
  index_t lo;
  index_t hi;
  device A* scratch;
};

template <class A>
struct GatherArgs {
  device const A* scratch;
};

template <class A>
struct ScratchArgs {
  device A* scratch;
};

constant int warp_size = 32;  // SIMD-group width of every Apple GPU family

/// Index of the contiguous run of `head` flags lane `lane` belongs to within its SIMD group:
/// lanes with equal ids form one segment. Every lane of the SIMD group must call it.
inline int segment_id(bool head, int lane) {
  const ulong heads = static_cast<ulong>(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(head)));
  const ulong upto = lane == 63 ? ~0ul : (2ul << lane) - 1ul;
  return metal::popcount(heads & upto);
}

// Euclidean norm in double-float in a fixed order (autan/kernels/norms.cuh on CUDA). Double-float
// keeps FP32's exponent range, so entries are first scaled by 2^k, exactly, with k set from the
// largest magnitude (max_by_chunk, scale_partials); then squares_by_chunk and sum_partials sum
// (2^k x)^2 by chunks of norm_block * norm_per_thread entries, each reduced by a tree in
// threadgroup memory, and the partials by one threadgroup. out[0] is the sum, out[1].hi is k.
// Launched with threadgroups of norm_block threads (autan::kernels::metal::sum_of_squares); the
// result depends on the data and its length only.
constant int norm_block = 256;
constant int norm_per_thread = 16;

template <class T>
struct NormData {
  device const T* x;
  device df64* partial;
  device df64* out;
  long n;
};

// Apple GPUs flush subnormal operands to zero, so the scaling reads the bits: the exponent of
// |x| (floor(log2 |x|), -1000 for zero, 1000 for inf and NaN) and 2^k x, exact for subnormals.
inline int exponent_of(float x) {
  const uint bits = as_type<uint>(x) & 0x7fffffffu;
  const uint e = bits >> 23;
  if (e == 255u) {
    return 1000;
  }
  if (e != 0u) {
    return int(e) - 127;
  }
  return bits == 0u ? -1000 : int(31u - metal::clz(bits)) - 149;
}
inline int exponent_of(df64 x) { return exponent_of(x.hi); }

inline float scaled_by(float x, int k) {
  const uint bits = as_type<uint>(x);
  if ((bits & 0x7f800000u) != 0u || (bits & 0x007fffffu) == 0u) {
    return metal::ldexp(x, k);
  }
  const float m = float(bits & 0x007fffffu);  // exact: 23 bits
  return metal::ldexp((bits >> 31) != 0u ? -m : m, k - 149);
}
inline df64 scaled_by(df64 x, int k) { return df64(scaled_by(x.hi, k), scaled_by(x.lo, k)); }

inline df64 square(float x) {
  const float p = x * x;
  return df64(p, fma(x, x, -p));
}
inline df64 square(df64 x) { return x * x; }

// Threadgroup memory holds float2 (hi, lo): df64's constructors do not apply in that space.
inline void tree_sum(threadgroup float2* s, uint t) {
  for (uint w = norm_block / 2; w > 0; w /= 2) {
    if (t < w) {
      const df64 v = df64(s[t].x, s[t].y) + df64(s[t + w].x, s[t + w].y);
      s[t] = float2(v.hi, v.lo);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
}

inline void tree_max(threadgroup int* s, uint t) {
  for (uint w = norm_block / 2; w > 0; w /= 2) {
    if (t < w) {
      s[t] = metal::max(s[t], s[t + w]);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
}

template <class T>
kernel void max_by_chunk(constant NormData<T>& d [[buffer(0)]],
                         uint t [[thread_position_in_threadgroup]],
                         uint g [[threadgroup_position_in_grid]]) {
  threadgroup int s[norm_block];
  const long base = long(g) * norm_block * norm_per_thread + long(t);
  int m = -1000;
  for (int j = 0; j < norm_per_thread; ++j) {
    const long i = base + long(j) * norm_block;
    if (i < d.n) {
      m = metal::max(m, exponent_of(d.x[i]));
    }
  }
  s[t] = m;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  tree_max(s, t);
  if (t == 0) {
    d.partial[g] = df64(float(s[0]));
  }
}

// k = -(largest exponent + 1), so 2^k max lies in [0.5, 1); 0 if every entry is zero or one is
// not finite (the sum then carries the infinity or NaN).
[[host_name("scale_partials_df64")]] kernel void scale_partials_df64(
    constant NormData<df64>& d [[buffer(0)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup int s[norm_block];
  int m = -1000;
  for (long i = long(t); i < d.n; i += norm_block) {
    m = metal::max(m, int(d.x[i].hi));
  }
  s[t] = m;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  tree_max(s, t);
  if (t == 0) {
    d.out[1] = df64(float(s[0] == -1000 || s[0] == 1000 ? 0 : -(s[0] + 1)));
  }
}

template <class T>
kernel void squares_by_chunk(constant NormData<T>& d [[buffer(0)]],
                             uint t [[thread_position_in_threadgroup]],
                             uint g [[threadgroup_position_in_grid]]) {
  threadgroup float2 s[norm_block];
  const int k = int(d.out[1].hi);
  const long base = long(g) * norm_block * norm_per_thread + long(t);
  df64 acc = df64(0.0f);
  for (int j = 0; j < norm_per_thread; ++j) {
    const long i = base + long(j) * norm_block;
    if (i < d.n) {
      acc = acc + square(scaled_by(d.x[i], k));
    }
  }
  s[t] = float2(acc.hi, acc.lo);
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  tree_sum(s, t);
  if (t == 0) {
    d.partial[g] = df64(s[0].x, s[0].y);
  }
}

[[host_name("sum_partials_df64")]] kernel void sum_partials_df64(
    constant NormData<df64>& d [[buffer(0)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float2 s[norm_block];
  df64 acc = df64(0.0f);
  for (long i = long(t); i < d.n; i += norm_block) {
    acc = acc + d.x[i];
  }
  s[t] = float2(acc.hi, acc.lo);
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  tree_sum(s, t);
  if (t == 0) {
    d.out[0] = df64(s[0].x, s[0].y);
  }
}

template [[host_name("max_by_chunk_float")]] kernel void max_by_chunk<float>(
    constant NormData<float>&, uint, uint);
template [[host_name("max_by_chunk_df64")]] kernel void max_by_chunk<df64>(
    constant NormData<df64>&, uint, uint);
template [[host_name("squares_by_chunk_float")]] kernel void squares_by_chunk<float>(
    constant NormData<float>&, uint, uint);
template [[host_name("squares_by_chunk_df64")]] kernel void squares_by_chunk<df64>(
    constant NormData<df64>&, uint, uint);

}  // namespace kernels
}  // namespace autan
