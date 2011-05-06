
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# The include_directories adds paths to both C/C++ compilation in the native
# compiler and cuda compilation in NVCC.  Note that CUDA_INCLUDE_DIRS is added
# automatically by CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.

# INCLUDE_DIRECTORIES(
#   ${CUDA_INCLUDE_DIRS}
#   )

##############################################################################
# There are four ways to compile source files with NVCC. 

# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# Use one executable only.
CUDA_ADD_EXECUTABLE( matrix_generator
  main.cu
  evo.cu
  matrix_copy.cu
  matrix_print.cu
  OPTIONS -arch=sm_13 -use_fast_math --ptxas-options=-v #-G3 -g -DDEBUG
#  OPTIONS -arch=sm_13 -use_fast_math --ptxas-options=-v -G3 -g -DDEBUG
#  DEBUG -G3 -g -DDEBUG
) 

# Or compile the cuda code into a shared library.

# Anything other than -D or /D is not passed along to nvcc.
add_definitions()

# You can set BUILD_SHARED_LIBS or you can pass STATIC, SHARED, or MODULE to
# CUDA_ADD_LIBRARY.  Remember that BUILD_SHARED_LIBS is only respected by
# CUDA_ADD_LIBRARY.  If you use CUDA_COMPILE or CUDA_WRAP_SRCS you must specify
# SHARED or STATIC as a parameter.
#set(BUILD_SHARED_LIBS ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#CUDA_ADD_LIBRARY(test_lib
#  test_lib.cu
#  external_dependency.h
#  # SHARED
#  # STATIC
#  OPTIONS "-DSTUFF=blah blah"
#  RELEASE --use_fast_math -DNDEBUG
#  DEBUG -g -DDEBUG
#  )

# Then link the shared library to another executable.
#ADD_EXECUTABLE( matrix_terminator
#  main.c
#  evo.cu
#)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

