cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("RainbowRayTracer" LANGUAGES CXX CUDA)

# Find and enable CUDA
find_package(CUDA REQUIRED)

# Add source to this project's executable.
add_executable (RainbowRayTracer "RainbowRayTracer.h" "RainbowRayTracer.cu")

# Set C++ and CUDA standard
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET RainbowRayTracer PROPERTY CXX_STANDARD 20)
  set_property(TARGET RainbowRayTracer PROPERTY CUDA_STANDARD 17)
endif()

# Enable CUDA separable compilation
set_target_properties(RainbowRayTracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(RainbowRayTracer ${CUDA_LIBRARIES})