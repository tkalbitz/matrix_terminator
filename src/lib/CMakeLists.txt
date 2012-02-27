# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

CUDA_ADD_LIBRARY( evo_matrix_generator
  evo_matrix_generator.cu
  ccpso2_matrix_generator.cu
  custom_matrix_generator.cu
  matrix_generator.h
  mat_lib_info.cu
  mat_lib_info.h
  evo_lib_instance.cu
  ccpso2_lib_instance.cu
  custom_lib_instance.cu
  ../ccpso2/pso_instance.cu
  ../ccpso2/pso_instance.h
  ../ccpso2/pso.cu
  ../ccpso2/pso.h
  ../ccpso2/pso_rating.cu
  ../ccpso2/pso_rating.h
  ../ccpso2/pso_setup.cu
  ../ccpso2/pso_setup.h
  ../ccpso2/pso_param_s.h
  ../ccpso2/pso_param_s.cu
  ../evo/instance.cu
  ../evo/instance.h
  ../evo/evo.cu
  ../evo/evo.h
  ../evo/evo_rating.cu
  ../evo/evo_setup.cu
  ../evo/matrix_copy.cu
  ../evo/matrix_copy.h
  ../custom/c_setup.cu
  ../custom/c_setup.h
  ../custom/c_rating.h
  ../custom/c_rating.cu
  ../custom/c_instance.cu
  ../custom/c_instance.h
  ../ya_malloc.h

  SHARED
  OPTIONS -arch=sm_13 -use_fast_math --ptxas-options=-v -ccbin ${CMAKE_CURRENT_SOURCE_DIR}/../../ccdir 
  RELEASE -O2 --compiler-options -Wall
  DEBUG -G3 -g -DDEBUG --maxrregcount=60
  )

ADD_EXECUTABLE( mat_lib_test
  lib_test.c
  matrix_generator.h
)  

add_dependencies(mat_lib_test evo_matrix_generator)
target_link_libraries(mat_lib_test evo_matrix_generator) 
