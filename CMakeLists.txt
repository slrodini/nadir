set(nadir_VERSION 1.0.0)

cmake_minimum_required(VERSION 3.10)

# Kill any build type specification, flags and optimizations from Oros
set(set CMAKE_BUILD_TYPE "Custom")

if(CMAKE_CONFIGURATION_TYPES)
   message(ERROR "Multi-config is not supported. Please specify build type explicitly
                  using -DCMAKE_BUILD_TYPE=<Release,Debug,RelWithDebInfo>")
endif()

project(Nadir CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
   -Wall -Wextra -Wpedantic
   -fPIC
)

find_program(OROS_CONFIG Oros-config REQUIRED)

if(OROS_CONFIG)
   execute_process(COMMAND ${OROS_CONFIG} --incdir OUTPUT_VARIABLE OROS_INC_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${OROS_CONFIG} --libdir OUTPUT_VARIABLE OROS_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${OROS_CONFIG} --allflags OUTPUT_VARIABLE OROS_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${OROS_CONFIG} --sanflags OUTPUT_VARIABLE OROS_SAN_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   set(OROS_INC_DIR ${OROS_INC_DIR} CACHE STRING INTERNAL)
   set(OROS_LIBRARIES ${OROS_LIBRARIES} CACHE STRING INTERNAL)
   separate_arguments(OROS_LIBRARIES_LIST UNIX_COMMAND "${OROS_LIBRARIES}")

   message(STATUS "Oros include directories: ${OROS_INC_DIR}")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OROS_FLAGS}")

   if(NOT "${OROS_SAN_FLAGS}" STREQUAL "")
      link_libraries(${OROS_SAN_FLAGS})
   endif()
else(OROS_CONFIG)
   message(FATAL_ERROR "Oros not found!")
endif(OROS_CONFIG)

find_package(Eigen3 3.3 REQUIRED)
if (Eigen3_FOUND)
message("-- Found Eigen version ${Eigen3_VERSION}: ${Eigen3_DIR}")
endif()

set(OROS_MODULES_HEADER_DIR "${PROJECT_BINARY_DIR}/Nadir")

file(MAKE_DIRECTORY "${OROS_MODULES_HEADER_DIR}")

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(MODULES_LIST)
set(includedir)
set(cflags)
set(libdir)
set(ldflags)

enable_testing()
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
file(GLOB source_files src/*.cc)

add_library(Nadir SHARED ${source_files})

target_link_directories(Nadir PUBLIC ${OROS_LIBRARIES_LIST})
target_link_libraries(Nadir PUBLIC Oros Eigen3::Eigen)


target_include_directories(Nadir PUBLIC ${PROJECT_SOURCE_DIR}/inc ${OROS_INC_DIR})

enable_testing()
add_subdirectory(tests)


