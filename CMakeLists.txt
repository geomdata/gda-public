cmake_minimum_required(VERSION 3.5...3.22)

project(gda-public)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

add_subdirectory(homology)
add_subdirectory(timeseries)
add_subdirectory(multidim)

