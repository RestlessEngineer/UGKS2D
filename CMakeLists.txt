cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 14)

project(UGKS2D)

set(EXECUTABLE_OUTPUT_PATH ./${CMAKE_BUILD_TYPE})

set(SOURCE src/main.cpp 
            src/tools.cpp 
            src/solver.cpp)

set(HEADER  src/global.h
			src/tools.h
            src/solver.h)

add_executable (${PROJECT_NAME} ${SOURCE})

