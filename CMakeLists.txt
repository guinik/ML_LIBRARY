cmake_minimum_required(VERSION 3.20)

option(USE_CUDA "Enable CUDA/cuBLAS acceleration for matrix multiplications" OFF)

if(USE_CUDA)
    project(ML_LIBRARY LANGUAGES CXX CUDA)
else()
    project(ML_LIBRARY LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ML_SOURCES
	src/Parameter.cpp
	src/Tensor.cpp
	src/MiniModel.cpp
	src/TransformerMiniModel.cpp
	src/Optimizer.cpp
	src/ExecutionGraph.cpp
	src/Node.cpp
	src/Layers.cpp
	src/Serialization.cpp
	src/DataLoader.cpp
)

if(USE_CUDA)
    list(APPEND ML_SOURCES src/CudaMatMul.cu)
    list(APPEND ML_SOURCES src/CudaOps.cu)
    list(APPEND ML_SOURCES src/CudaOptimizer.cu)
endif()

set(ML_HEADERS
	include/Parameter.hpp
	include/Tensor.hpp
	include/MiniModel.hpp
	include/TransformerMiniModel.hpp
	include/Optimizer.hpp
	include/ExecutionGraph.hpp
	include/Node.hpp
	include/Layers.hpp
	include/Serialization.hpp
	include/DataLoader.hpp
	include/CudaMatMul.hpp
	include/CudaOps.hpp
)



source_group("Sources" FILES ${ML_SOURCES})
source_group("Headers" FILES ${ML_HEADERS})


add_library(ML_LIBRARY STATIC ${ML_SOURCES} ${ML_HEADERS})
if(MSVC)
    target_compile_options(ML_LIBRARY PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)
else()
    target_compile_options(ML_LIBRARY PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -O3>)
endif()

target_include_directories(ML_LIBRARY PUBLIC include)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(ML_LIBRARY PUBLIC USE_CUDA)
    target_link_libraries(ML_LIBRARY PUBLIC CUDA::cublas CUDA::cudart)
    set_target_properties(ML_LIBRARY PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(EXECUTE src/main.cpp)

if(NOT MSVC)
    target_compile_options(EXECUTE PRIVATE -O3)
endif()

target_link_libraries(EXECUTE PRIVATE ML_LIBRARY)