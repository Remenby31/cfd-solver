add_executable(autan_solvers_tests test_multigrid.cpp)
target_link_libraries(
  autan_solvers_tests
  PRIVATE autan::solvers autan::io autan::partition Catch2::Catch2WithMain autan_compile_options
)
target_compile_definitions(
  autan_solvers_tests
  PRIVATE AUTAN_TEST_MESH_DIR="${PROJECT_SOURCE_DIR}/tests/data/mesh"
)
catch_discover_tests(autan_solvers_tests)
