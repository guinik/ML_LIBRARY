cmake_minimum_required(VERSION 3.20)
project(ML_LIBRARY)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ML_SOURCES
	src/Parameter.cpp
	src/Matrix.cpp
	src/MiniModel.cpp
)

set(ML_HEADERS
	include/Parameter.hpp
	include/Matrix.hpp
	include/MiniModel.hpp

)


source_group("Sources" FILES ${ML_SOURCES})
source_group("Headers" FILES ${ML_HEADERS})


add_library(ML_LIBRARY STATIC ${ML_SOURCES} ${ML_HEADERS})
if(MSVC)
    target_compile_options(ML_LIBRARY PRIVATE /W4 /WX)
else()
    target_compile_options(ML_LIBRARY PRIVATE -Wall -Wextra -Werror)
endif()

target_include_directories(ML_LIBRARY PUBLIC include)

add_executable(EXECUTE src/main.cpp)

target_link_libraries(EXECUTE PRIVATE ML_LIBRARY)