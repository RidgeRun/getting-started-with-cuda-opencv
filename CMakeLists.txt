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

add_executable (sobel_gpu_1_naive sobel_gpu_1_naive.cc)
target_link_libraries (sobel_gpu_1_naive ${OpenCV_LIBS})

add_executable (sobel_gpu_2_single_alloc sobel_gpu_2_single_alloc.cc)
target_link_libraries (sobel_gpu_2_single_alloc ${OpenCV_LIBS})
