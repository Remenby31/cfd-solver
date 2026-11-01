FetchContent_GetProperties(picosha2)

add_executable(autan-run-relaxation relaxation_main.cpp)
target_link_libraries(autan-run-relaxation PRIVATE autan::runtime autan_compile_options)

add_executable(autan_runtime_tests test_runtime.cpp)
target_include_directories(autan_runtime_tests SYSTEM PRIVATE ${picosha2_SOURCE_DIR})
target_link_libraries(
  autan_runtime_tests
  PRIVATE
    autan::runtime
    autan::physics_runtime
    autan_cgns
    nlohmann_json::nlohmann_json
    Catch2::Catch2WithMain
    autan_compile_options
)
target_compile_definitions(
  autan_runtime_tests
  PRIVATE
    AUTAN_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
    AUTAN_RUN_RELAXATION="$<TARGET_FILE:autan-run-relaxation>"
)
add_dependencies(autan_runtime_tests autan-run-relaxation)
if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  foreach(target autan_runtime_tests autan-run-relaxation)
    target_compile_definitions(${target} PRIVATE AUTAN_HAVE_CUDA)
    target_link_libraries(${target} PRIVATE CUDA::cudart)
  endforeach()
endif()
if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
  target_compile_definitions(
    autan_runtime_tests
    PRIVATE
      AUTAN_MPIEXEC="${MPIEXEC_EXECUTABLE}"
      AUTAN_MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG}"
  )
endif()
# The multi-rank tests carry the label `mpi` (ctest -L mpi).
catch_discover_tests(autan_runtime_tests TEST_SPEC "~[mpi]~[gpu]")
catch_discover_tests(autan_runtime_tests TEST_SPEC "[gpu]~[mpi]" PROPERTIES LABELS gpu)
catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi]~[gpu]" PROPERTIES LABELS mpi)
# [mpi][gpu]: on several ranks, and on the GPU too in a CUDA build (its tags as labels).
if(AUTAN_ENABLE_CUDA)
  catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi][gpu]" ADD_TAGS_AS_LABELS)
else()
  catch_discover_tests(autan_runtime_tests TEST_SPEC "[mpi][gpu]" PROPERTIES LABELS mpi)
endif()

# Ranks through the runtime's communicator: one rank here, 1 to 4 under MPI.
add_executable(autan_ranks_tests test_ranks.cpp)
target_include_directories(autan_ranks_tests PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(autan_ranks_tests PRIVATE autan::runtime Catch2::Catch2 autan_compile_options)
add_test(NAME autan_ranks_tests COMMAND autan_ranks_tests)
if(AUTAN_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(autan_ranks_tests PRIVATE AUTAN_HAVE_MPI)
  target_link_libraries(autan_ranks_tests PRIVATE MPI::MPI_CXX)
  foreach(ranks 2 4)
    add_test(
      NAME autan_ranks_tests_${ranks}_ranks
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:autan_ranks_tests> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
      autan_ranks_tests_${ranks}_ranks
      PROPERTIES PROCESSORS ${ranks} TIMEOUT 300 LABELS mpi
    )
  endforeach()
endif()

add_executable(autan_exact_sum_tests test_exact_sum.cpp)
target_link_libraries(
  autan_exact_sum_tests
  PRIVATE autan::runtime Catch2::Catch2WithMain autan_compile_options
)
catch_discover_tests(autan_exact_sum_tests)

if(AUTAN_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # The kernel launch, compiled by nvcc without the host warning flags.
  add_library(autan_exact_sum_device OBJECT exact_sum_device.cu)
  set_target_properties(
    autan_exact_sum_device
    PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(autan_exact_sum_device PRIVATE autan::runtime CUDA::cudart)
  add_executable(
    autan_exact_sum_device_tests
    test_exact_sum_device.cpp
    $<TARGET_OBJECTS:autan_exact_sum_device>
  )
  target_link_libraries(
    autan_exact_sum_device_tests
    PRIVATE autan::runtime CUDA::cudart Catch2::Catch2WithMain autan_compile_options
  )
  catch_discover_tests(autan_exact_sum_device_tests PROPERTIES LABELS gpu)
endif()

# autan-run as shipped, through its output and exit status: no physics yet.
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/data/runtime/hybrid.solver-input.json
)
file(READ ${PROJECT_SOURCE_DIR}/tests/data/runtime/hybrid.solver-input.json shipped_input)
string(
  REPLACE "@MESH@"
  "${PROJECT_SOURCE_DIR}/tests/data/mesh/hybrid/polyMesh"
  shipped_input
  "${shipped_input}"
)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hybrid.solver-input.json "${shipped_input}")
add_test(
  NAME autan-run.no-physics
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-run>
    "-DARGS=--input;${CMAKE_CURRENT_BINARY_DIR}/hybrid.solver-input.json;--runs;${CMAKE_CURRENT_BINARY_DIR}/runs"
    -DEXPECT_CODE=2 "-DEXPECT=E0706" -P ${PROJECT_SOURCE_DIR}/tests/io/run_cli.cmake
)
add_test(
  NAME autan-run.usage
  COMMAND
    ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:autan-run> -DARGS=--bogus -DEXPECT_CODE=2
    "-DEXPECT=usage: autan-run" -P ${PROJECT_SOURCE_DIR}/tests/io/run_cli.cmake
)
