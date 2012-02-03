# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../evo
  )

CUDA_ADD_LIBRARY( evo_matrix_generator
  evo_matrix_generator.cu
  pso_matrix_generator.cu
  matrix_generator.h
  mat_lib_info.cu
  mat_lib_info.h
  evo_lib_instance.cu
  pso_lib_instance.cu
  ../pso/pso_instance.cu
  ../pso/pso_instance.h
  ../pso/pso.cu
  ../pso/pso.h
  ../pso/pso_rating.cu
  ../pso/pso_rating.h
  ../pso/pso_setup.cu
  ../pso/pso_setup.h
  ../pso/pso_param_s.h
  ../pso/pso_param_s.cu
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
  lib_test.c
  matrix_generator.h
)  

add_dependencies(evo_lib_test evo_matrix_generator)
target_link_libraries(evo_lib_test evo_matrix_generator) 
