# Linear solvers of the implicit step: agglomeration multigrid with line-implicit smoothing.
add_library(autan_solvers src/agglomeration.cpp src/gcr.cpp src/line_smoother.cpp src/multigrid.cpp)
add_library(autan::solvers ALIAS autan_solvers)
target_include_directories(
  autan_solvers
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(autan_solvers PUBLIC autan::core PRIVATE autan_compile_options)
set_target_properties(autan_solvers PROPERTIES POSITION_INDEPENDENT_CODE ON)
