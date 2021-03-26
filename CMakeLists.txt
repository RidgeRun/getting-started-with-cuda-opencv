cmake_minimum_required (VERSION 2.8)
cmake_policy(SET CMP0048 NEW)

# Initialize project
project (GettingStartedWithOpenCVCuda LANGUAGES CXX VERSION 0.1.0)

# Dependencies
find_package (OpenCV REQUIRED)

# Overlay include search path
include_directories (${OpenCV_INCLUDE_DIRS})

# Programs
add_executable (sobel_cpu sobel_cpu.cc)
target_link_libraries (sobel_cpu ${OpenCV_LIBS})
