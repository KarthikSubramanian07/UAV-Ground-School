cmake_minimum_required(VERSION 3.16)
project(week02_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Works with OpenCV 4.x (apt libopencv-dev) and 5.x (Homebrew).
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

foreach(target color_me_impressed needin_stitches)
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra)
  endif()
endforeach()
