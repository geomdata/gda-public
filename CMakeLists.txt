cmake_minimum_required(VERSION 3.5...3.22)

project(gda-public)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
message(STATUS "Python_NumPy_FOUND: ${Python_NumPy_FOUND}")
message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "Python_NumPy_VERSION: ${Python_NumPy_VERSION}")

# For some reason, cmake isn't using those paths.  Force them.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${Python_NumPy_INCLUDE_DIRS} -L${Python_NumPy_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${Python_NumPy_INCLUDE_DIRS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${Python_NumPy_INCLUDE_DIRS}")

add_subdirectory(homology)
add_subdirectory(timeseries)
add_subdirectory(multidim)
