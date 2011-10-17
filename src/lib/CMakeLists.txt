# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../evo
  )

CUDA_ADD_LIBRARY( evo_matrix_generator
  matrix_generator.cu
  matrix_generator.h
  evo_info.cu
  evo_info.h
  evo_instance.cu
  ../evo/instance.cu
  ../evo/instance.h
  ../evo/evo.cu
  ../evo/evo.h
  ../evo/evo_rating.cu
  ../evo/evo_setup.cu
  ../evo/matrix_copy.cu
  ../evo/matrix_copy.h
  ../ya_malloc.h

  SHARED
  OPTIONS -arch=sm_13 -use_fast_math --ptxas-options=-v -ccbin ${CMAKE_CURRENT_SOURCE_DIR}/../../ccdir 
  RELEASE -O2 --compiler-options -Wall
  DEBUG -G3 -g -DDEBUG --maxrregcount=60
  )

ADD_EXECUTABLE( evo_lib_test
  evo_lib_test.c
  matrix_generator.h
)  

add_dependencies(evo_lib_test evo_matrix_generator)
target_link_libraries(evo_lib_test evo_matrix_generator) 
