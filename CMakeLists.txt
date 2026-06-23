cmake_minimum_required(VERSION 3.20)
project(ML_LIBRARY)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ML_SOURCES
	src/Parameter.cpp
	src/Tensor.cpp
	src/MiniModel.cpp
	src/Optimizer.cpp
	src/ExecutionGraph.cpp
	src/Node.cpp
	src/Layers.cpp
)

set(ML_HEADERS
	include/Parameter.hpp
	include/Tensor.hpp
	include/MiniModel.hpp
	include/Optimizer.hpp
	include/ExecutionGraph.hpp
	include/Node.hpp
	include/Layers.hpp
)


source_group("Sources" FILES ${ML_SOURCES})
source_group("Headers" FILES ${ML_HEADERS})


add_library(ML_LIBRARY STATIC ${ML_SOURCES} ${ML_HEADERS})
if(MSVC)
    target_compile_options(ML_LIBRARY PRIVATE /W4 /WX)
else()
    target_compile_options(ML_LIBRARY PRIVATE -Wall -Wextra -Werror -O3)
endif()

target_include_directories(ML_LIBRARY PUBLIC include)

add_executable(EXECUTE src/main.cpp)

if(NOT MSVC)
    target_compile_options(EXECUTE PRIVATE -O3)
endif()

target_link_libraries(EXECUTE PRIVATE ML_LIBRARY)