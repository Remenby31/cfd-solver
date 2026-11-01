add_executable(autan_physics_tests test_euler.cpp test_implicit.cpp test_manufactured.cpp)
target_link_libraries(
  autan_physics_tests
  PRIVATE autan::physics autan::io Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_physics_tests
  PRIVATE
    AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
    AUTAN_EULER_MMS="${PROJECT_SOURCE_DIR}/py/tests/fixtures/euler.mms.json"
)
catch_discover_tests(autan_physics_tests)

# The Euler residual on the generated Metal loops against the CPU physics (macOS).
if(AUTAN_ENABLE_METAL)
  enable_language(OBJCXX)
  add_executable(autan_physics_metal_tests test_euler_metal.mm)
  target_compile_options(autan_physics_metal_tests PRIVATE -fobjc-arc)
  target_include_directories(
    autan_physics_metal_tests
    PRIVATE ${PROJECT_BINARY_DIR}/physics/generated
  )
  target_link_libraries(
    autan_physics_metal_tests
    PRIVATE autan::physics autan::io autan_metal Catch2::Catch2WithMain autan_compile_options
  )
  target_compile_definitions(
    autan_physics_metal_tests
    PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
  )
  catch_discover_tests(autan_physics_metal_tests)
endif()

# The Euler physics through autan-run's run_case; the runtime needs CGNS.
if(AUTAN_ENABLE_CGNS)
  FetchContent_GetProperties(picosha2)
  add_executable(autan_euler_run_tests test_euler_run.cpp)
  target_include_directories(autan_euler_run_tests SYSTEM PRIVATE ${picosha2_SOURCE_DIR})
  target_link_libraries(
    autan_euler_run_tests
    PRIVATE
      autan::physics_runtime
      autan_cgns
      nlohmann_json::nlohmann_json
      Catch2::Catch2WithMain
      autan_compile_options
  )
  target_compile_definitions(
    autan_euler_run_tests
    PRIVATE AUTAN_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
  )
  # Several ranks: autan-run under the MPI launcher, as tests/runtime does.
  if(AUTAN_ENABLE_MPI AND AUTAN_ENABLE_METIS)
    target_compile_definitions(
      autan_euler_run_tests
      PRIVATE
        AUTAN_RUN="$<TARGET_FILE:autan-run>"
        AUTAN_MPIEXEC="${MPIEXEC_EXECUTABLE}"
        AUTAN_MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG}"
    )
    add_dependencies(autan_euler_run_tests autan-run)
  endif()
  # The multi-rank tests carry the label `mpi` (ctest -L mpi), the device ones `gpu`.
  catch_discover_tests(autan_euler_run_tests TEST_SPEC "~[gpu]~[mpi]")
  catch_discover_tests(autan_euler_run_tests TEST_SPEC "[mpi]" PROPERTIES LABELS mpi)
  if(AUTAN_ENABLE_CUDA)
    target_link_libraries(autan_euler_run_tests PRIVATE autan::physics)
    catch_discover_tests(autan_euler_run_tests TEST_SPEC "[gpu]" PROPERTIES LABELS gpu)
  endif()
endif()

# The Euler residual on a CUDA device against the host's.
if(AUTAN_ENABLE_CUDA AND AUTAN_ENABLE_CGNS)
  add_executable(autan_euler_device_tests test_euler_device.cpp)
  target_link_libraries(
    autan_euler_device_tests
    PRIVATE
      autan::physics_runtime
      autan::physics
      autan::io
      Catch2::Catch2WithMain
      autan_compile_options
  )
  target_compile_definitions(
    autan_euler_device_tests
    PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
  )
  catch_discover_tests(autan_euler_device_tests PROPERTIES LABELS gpu)
endif()
