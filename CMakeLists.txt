cmake_minimum_required(VERSION 2.8.15)

set (CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

project(UGKS2D)

option(BUILD_EXAMPLES "Whether to build an examples or not." ON)


set(EXECUTABLE_OUTPUT_PATH ./${CMAKE_BUILD_TYPE})

set(SOURCE src/main.cpp 
            src/tools.cpp 
            src/solver.cpp)

set(HEADER  src/global.h
			src/tools.h
            src/solver.h)


add_executable (${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME} gomp)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()