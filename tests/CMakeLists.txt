file(GLOB TEST_SOURCES "*.cc")

# configure_file(
#    ${CMAKE_CURRENT_SOURCE_DIR}/test_config.hpp.in
#    ${CMAKE_CURRENT_BINARY_DIR}/test_config.hpp
#    @ONLY
# )

foreach(TEST_SOURCE ${TEST_SOURCES})
   get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
   add_executable(${TEST_NAME} ${TEST_SOURCE})
   target_include_directories(${TEST_NAME} PRIVATE
      ${PROJECT_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}
      ${PROJECT_SOURCE_DIR}/inc
   )

   target_link_libraries(${TEST_NAME}
      Nadir
   )

   # Add test
   add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()