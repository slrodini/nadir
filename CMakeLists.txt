set(nadir_VERSION 1.0.0)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)

if(CMAKE_CONFIGURATION_TYPES)
   message(ERROR "Multi-config is not supported. Please specify build type explicitly
                  using -DCMAKE_BUILD_TYPE=<Release,Debug,RelWithDebInfo>")
endif()

project(Nadir CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
   message(STATUS "No build type specified. Defaulting to Release.")
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_VECTORIZATION "Enable vectorization flags" ON)
if(ENABLE_VECTORIZATION)
   message(STATUS "Vectorization flags will be applied")
else()
   message(STATUS "Vectorization flags will be disabled")
endif()

add_compile_options(
   -Wall -Wextra -Wpedantic
   -fPIC
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
   if(CMAKE_BUILD_TYPE STREQUAL "Release")
      if(ENABLE_VECTORIZATION)
         message(STATUS "Applying SSE2 flags for x86_64 architecture")
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=sse -msse2")
         set(NADIR_VEC_FLAGS "${NADIR_VEC_FLAGS} -mfpmath=sse -msse2")
      endif()
   endif()
endif()

# The leak sanitizer is not supported on arm64
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
   set(SANITIZER_FLAGS "-fsanitize=address,undefined,alignment,bounds,null,vptr")

   if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(SANITIZER_FLAGS "${SANITIZER_FLAGS},leak")
   endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -DDEBUG ${SANITIZER_FLAGS}")
   link_libraries(${SANITIZER_FLAGS})
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -funroll-loops")
   if(ENABLE_VECTORIZATION)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
      set(NADIR_VEC_FLAGS "${NADIR_VEC_FLAGS} -mtune=native -march=native")
   endif()

elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -DDEBUG ${SANITIZER_FLAGS}")
   if(ENABLE_VECTORIZATION)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
      set(NADIR_VEC_FLAGS "${NADIR_VEC_FLAGS} -mtune=native -march=native")
   endif()

   link_libraries(${SANITIZER_FLAGS})

else()
   message(STATUS "Build type: ${CMAKE_BUILD_TYPE} not handled directly, resorting to Debug.")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -DDEBUG ${SANITIZER_FLAGS}")
   link_libraries(${SANITIZER_FLAGS})
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build type specific CXX Flags: ${CMAKE_CXX_FLAGS}")

if(NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()


find_package(Eigen3 3.3 REQUIRED)
if (Eigen3_FOUND)
   message("-- Found Eigen version ${Eigen3_VERSION}: ${Eigen3_DIR}")
endif()

set(NADIR_HEADER_DIR "${PROJECT_BINARY_DIR}/Nadir")

file(MAKE_DIRECTORY "${NADIR_HEADER_DIR}")

file(GLOB source_files src/*.cc)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(Nadir SHARED ${source_files})


target_link_libraries(Nadir PUBLIC Eigen3::Eigen)


# target_include_directories(Nadir PUBLIC ${PROJECT_SOURCE_DIR}/inc)
target_include_directories(Nadir
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)


enable_testing()
add_subdirectory(tests)

set(prefix ${CMAKE_INSTALL_PREFIX})
set(includedir "${prefix}/include")
set(libdir "${prefix}/lib")
set(cflags "-I${prefix}/include")
set(ldflags "-L${prefix}/lib -lNadir")
set(allflags "${CMAKE_CXX_FLAGS}")
set(vecflags "${NADIR_VEC_FLAGS}")
set(sanflags "${SANITIZER_FLAGS}")
configure_file("${PROJECT_SOURCE_DIR}/bin/Nadir-config.in" "${PROJECT_BINARY_DIR}/bin/Nadir-config")

# install(TARGETS Nadir RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
install(TARGETS Nadir
    EXPORT NadirTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY inc/nadir DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/bin/Nadir-config DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(EXPORT NadirTargets
    FILE NadirTargets.cmake
    NAMESPACE Nadir::
    DESTINATION lib/cmake/Nadir
)


include(CMakePackageConfigHelpers)

# Generate version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NadirConfigVersion.cmake"
    VERSION ${nadir_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NadirConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NadirConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Nadir
)

# Install the package config and version files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NadirConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NadirConfigVersion.cmake"
    DESTINATION lib/cmake/Nadir
)

# ============================
# DOCUMENTATION
# ============================
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
   message(STATUS "Found Doxygen: ${DOXYGEN_EXECUTABLE}")

   set(DOXYGEN_DIR ${CMAKE_SOURCE_DIR}/docs/doxygen)
   set(DOXYFILE ${DOXYGEN_DIR}/Doxyfile)
   set(DOC_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/docs)

   add_custom_target(Nadir-doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
      WORKING_DIRECTORY ${DOXYGEN_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
   )
else()
   message(WARNING "Doxygen not found in the system! Cannot generate documentation.")
endif()