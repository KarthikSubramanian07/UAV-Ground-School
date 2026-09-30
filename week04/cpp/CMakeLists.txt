cmake_minimum_required(VERSION 3.16)
project(week04_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Standard library only: no OpenCV, no crypto library (SHA-256 is in protocols.cpp).
add_executable(protocol_tool protocol_tool.cpp protocols.cpp)
if(MSVC)
  target_compile_options(protocol_tool PRIVATE /W4)
else()
  # No fused multiply add, so x * 1e7 before round() rounds exactly like Python's float math.
  target_compile_options(protocol_tool PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
