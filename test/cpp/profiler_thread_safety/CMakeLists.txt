cmake_minimum_required(VERSION 3.18)
project(profiler_upload_thread_safety LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "These tests use the real Linux pthread/prctl profiler Thread implementation")
endif()

get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
find_package(Threads REQUIRED)
set(PROFILER_SANITIZER "none" CACHE STRING "none, address, or thread")
set_property(CACHE PROFILER_SANITIZER PROPERTY STRINGS none address thread)

add_executable(profiler_upload_test
  profiler_thread_safety_test.cpp
  "${REPOSITORY_ROOT}/torch_npu/csrc/toolkit/profiler/src/data_dumper.cpp")
target_compile_features(profiler_upload_test PRIVATE cxx_std_17)
target_include_directories(profiler_upload_test PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/stubs" "${REPOSITORY_ROOT}")
target_link_libraries(profiler_upload_test PRIVATE Threads::Threads)
if(PROFILER_SANITIZER STREQUAL "address")
  target_compile_options(profiler_upload_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(profiler_upload_test PRIVATE -fsanitize=address,undefined)
elseif(PROFILER_SANITIZER STREQUAL "thread")
  target_compile_options(profiler_upload_test PRIVATE -fsanitize=thread -g)
  target_link_options(profiler_upload_test PRIVATE -fsanitize=thread)
elseif(NOT PROFILER_SANITIZER STREQUAL "none")
  message(FATAL_ERROR "Unknown PROFILER_SANITIZER: ${PROFILER_SANITIZER}")
endif()

enable_testing()
add_test(NAME profiler_upload.rb64 COMMAND profiler_upload_test rb 8 50000 64)
add_test(NAME profiler_upload.rb1024 COMMAND profiler_upload_test rb 8 50000 1024)
add_test(NAME profiler_upload.dumper COMMAND profiler_upload_test dumper "${CMAKE_CURRENT_BINARY_DIR}/dumper-output" 6 100 1024)
add_test(NAME profiler_upload.trace COMMAND profiler_upload_test trace "${CMAKE_CURRENT_BINARY_DIR}/trace-output" 6 100 64)
set_tests_properties(profiler_upload.rb64 profiler_upload.rb1024
  profiler_upload.dumper profiler_upload.trace PROPERTIES TIMEOUT 180)
