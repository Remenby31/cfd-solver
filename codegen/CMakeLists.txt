# Kernel generator: runs autan_codegen on kernels/*.py at build time and builds the generated loops.
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

set(
  AUTAN_KERNEL_MODULES
  advection
  advection_diffusion
  blocks
  gradients
  roe
)
# AUTAN_ENABLE_CUDA (root) adds the GPU loops, the GPU spike driver and the loop benchmark.
# AUTAN_ENABLE_HIP compiles the same generated GPU loops with HIP (no AMD device: compile only).
option(AUTAN_ENABLE_HIP "Compile the generated GPU loops with HIP" OFF)
# AUTAN_ENABLE_METAL adds the Metal loops (Apple GPUs), compiled from source at first use.
option(AUTAN_ENABLE_METAL "Build the Metal loops and runtime (macOS)" OFF)
if(AUTAN_ENABLE_METAL AND NOT APPLE)
  message(FATAL_ERROR "AUTAN_ENABLE_METAL needs macOS")
endif()
if(AUTAN_ENABLE_CUDA AND AUTAN_ENABLE_HIP)
  message(FATAL_ERROR "AUTAN_ENABLE_CUDA and AUTAN_ENABLE_HIP are exclusive")
endif()
set(AUTAN_KERNEL_BACKENDS cpu)
if(AUTAN_ENABLE_CUDA OR AUTAN_ENABLE_HIP)
  list(APPEND AUTAN_KERNEL_BACKENDS gpu)
endif()
if(AUTAN_ENABLE_METAL)
  list(APPEND AUTAN_KERNEL_BACKENDS metal)
endif()

file(
  GLOB AUTAN_CODEGEN_SOURCES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/autan_codegen/*.py
)
set(AUTAN_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

set(AUTAN_GENERATED_FILES)
set(AUTAN_GENERATED_GPU_UNITS)
# Test kernels (codegen/tests/kernels) are built into the spike library and the loop benchmark only.
set(AUTAN_TEST_KERNEL_MODULES diagonal_only face_weighted)
# Modules with bodies computing in the accumulation type, which Metal cannot divide in (E1013).
set(AUTAN_ACCUM_MODULES blocks diagonal_only)
foreach(module IN LISTS AUTAN_KERNEL_MODULES AUTAN_TEST_KERNEL_MODULES)
  if(module IN_LIST AUTAN_TEST_KERNEL_MODULES)
    set(module_source ${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/${module}.py)
  else()
    set(module_source ${PROJECT_SOURCE_DIR}/kernels/${module}.py)
  endif()
  set(module_backends ${AUTAN_KERNEL_BACKENDS})
  if(module IN_LIST AUTAN_ACCUM_MODULES)
    list(REMOVE_ITEM module_backends metal)
  endif()
  set(outputs ${AUTAN_GENERATED_DIR}/${module}.hpp)
  list(APPEND outputs ${AUTAN_GENERATED_DIR}/${module}_cpu.hpp)
  if(AUTAN_ENABLE_CUDA OR AUTAN_ENABLE_HIP)
    list(
      APPEND outputs
      ${AUTAN_GENERATED_DIR}/${module}_gpu.cuh
      ${AUTAN_GENERATED_DIR}/${module}_gpu.cu
    )
    list(APPEND AUTAN_GENERATED_GPU_UNITS ${AUTAN_GENERATED_DIR}/${module}_gpu.cu)
  endif()
  if("metal" IN_LIST module_backends)
    list(APPEND outputs ${AUTAN_GENERATED_DIR}/${module}_metal.hpp)
  endif()
  list(JOIN module_backends "," backends)
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src ${Python3_EXECUTABLE} -m
      autan_codegen ${module_source} --out ${AUTAN_GENERATED_DIR} --backends ${backends}
    DEPENDS
      ${module_source}
      ${AUTAN_CODEGEN_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/autan_codegen/metal_prelude.metal
    COMMENT "Generating kernels from ${module}.py"
    VERBATIM
  )
  list(APPEND AUTAN_GENERATED_FILES ${outputs})
endforeach()
add_custom_target(autan_generated_kernels DEPENDS ${AUTAN_GENERATED_FILES})

add_library(autan_kernels cpp/geometry.cpp cpp/colouring.cpp)
add_library(autan::kernels ALIAS autan_kernels)
add_dependencies(autan_kernels autan_generated_kernels)
target_include_directories(
  autan_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${AUTAN_GENERATED_DIR}>
)
target_link_libraries(autan_kernels PUBLIC autan::core PRIVATE autan_compile_options)
set_target_properties(autan_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The spike drivers are shared libraries loaded by ctypes, so the static core they link must be
# position independent.
set_target_properties(autan_core autan_io autan_partition PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Spike driver: C ABI over the generated host loops, loaded by codegen/tests through ctypes.
add_library(autan_spike SHARED spike/capi.cpp)
target_link_libraries(
  autan_spike
  PRIVATE autan::kernels autan::io autan::partition autan_compile_options
)

if(AUTAN_ENABLE_CUDA OR AUTAN_ENABLE_METAL)
  # Commit of the benchmark binaries, for the provenance autan-bench records.
  find_package(Git QUIET)
  add_custom_target(
    autan_codegen_revision
    COMMAND
      ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
      "-DAUTAN_COMMIT=${AUTAN_COMMIT}" -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/revision/revision.h
      -P ${PROJECT_SOURCE_DIR}/bench/cmake/revision.cmake
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision/revision.h
  )
endif()

if(AUTAN_ENABLE_CUDA)
  # Every generated loop instantiated for every policy: the compile-time measure of the spike.
  add_library(autan_kernels_gpu OBJECT ${AUTAN_GENERATED_GPU_UNITS})
  target_link_libraries(autan_kernels_gpu PRIVATE autan::kernels)
  set_target_properties(autan_kernels_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

  # GPU spike driver: the same C ABI on device memory, and the loop benchmark.
  add_library(autan_spike_gpu SHARED spike/capi.cpp spike/capi_gpu.cu)
  target_link_libraries(autan_spike_gpu PRIVATE autan::kernels autan::io autan::partition)
  set_target_properties(autan_spike_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(
    autan_spike_gpu
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xptxas=-warn-spills>
  )

  add_executable(autan_loop_bench bench/loop_bench.cu spike/capi.cpp)
  add_dependencies(autan_loop_bench autan_codegen_revision)
  target_include_directories(autan_loop_bench PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/revision spike)
  target_link_libraries(autan_loop_bench PRIVATE autan::kernels autan::io autan::partition)
  set_target_properties(autan_loop_bench PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(autan_loop_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v>)

  # Option B of research section 3: the Roe Jacobian hand-written on dual numbers.
  add_executable(autan_option_b bench/option_b.cu)
  target_link_libraries(autan_option_b PRIVATE autan::kernels autan::io)
  set_target_properties(autan_option_b PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(autan_option_b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

if(AUTAN_ENABLE_HIP)
  enable_language(HIP)
  set_source_files_properties(${AUTAN_GENERATED_GPU_UNITS} PROPERTIES LANGUAGE HIP)
  add_library(autan_kernels_gpu OBJECT ${AUTAN_GENERATED_GPU_UNITS})
  add_dependencies(autan_kernels_gpu autan_generated_kernels)
  target_include_directories(
    autan_kernels_gpu
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${AUTAN_GENERATED_DIR}
  )
  set_target_properties(autan_kernels_gpu PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
endif()

if(AUTAN_ENABLE_METAL)
  enable_language(OBJCXX)
  add_library(autan_metal cpp/metal.mm)
  target_include_directories(
    autan_metal
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_compile_options(autan_metal PRIVATE -fobjc-arc)
  target_compile_features(autan_metal PUBLIC cxx_std_20)
  target_link_libraries(
    autan_metal
    PUBLIC "-framework Metal" "-framework Foundation" "-framework IOKit"
    PRIVATE autan_compile_options
  )
  set_target_properties(autan_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Metal spike driver: the C ABI of the CUDA driver on the generated Metal loops.
  add_library(autan_spike_metal SHARED spike/capi.cpp spike/capi_metal.mm)
  target_compile_options(autan_spike_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_include_directories(autan_spike_metal PRIVATE spike)
  target_link_libraries(
    autan_spike_metal
    PRIVATE autan::kernels autan::io autan::partition autan_metal autan_compile_options
  )

  # Peak bandwidth of the Mac's GPU, and the generated loops timed against it.
  add_executable(autan_metal_stream bench/metal_stream.mm)
  add_dependencies(autan_metal_stream autan_codegen_revision)
  target_include_directories(
    autan_metal_stream
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/revision spike bench
  )
  target_compile_options(autan_metal_stream PRIVATE -fobjc-arc)
  target_link_libraries(autan_metal_stream PRIVATE autan_metal)

  add_executable(autan_metal_loop_bench bench/metal_loop_bench.mm spike/capi.cpp)
  add_dependencies(autan_metal_loop_bench autan_codegen_revision)
  target_include_directories(
    autan_metal_loop_bench
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/revision spike bench
  )
  target_compile_options(autan_metal_loop_bench PRIVATE -fobjc-arc)
  target_link_libraries(
    autan_metal_loop_bench
    PRIVATE autan::kernels autan::io autan::partition autan_metal
  )
endif()
