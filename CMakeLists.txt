cmake_minimum_required(VERSION 3.10)

project(UGKS2D)

set(SOURCES src/main.cpp 
            src/tools.cpp 
            src/solver.cpp)

set(HEADERS  src/global.h
			 src/tools.h
             src/solver.h)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found. Version: ${Eigen3_VERSION}")
    include_directories(${EIGEN3_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Eigen3 not found. Please install it or specify its location.")
endif()

if(OpenMP_FOUND)
    message(STATUS "OpenMP found. Version: ${OpenMP_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found. Please install it or specify its location.")
endif()

#
# OPTIONS
#
option(TRANSFER_BUILD_EXAMPLES "Whether to build an examples or not." ON)
option(USE_OPENMP "Whether to use openMP or not" ON)
option(USE_OPENMP_PROFILIZE "Whether to use profilizing or not" OFF)

set(EXECUTABLE_OUTPUT_PATH ./${CMAKE_BUILD_TYPE})

if(USE_OPENMP_PROFILIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDO_PROFILIZE")
endif()

#
# ArgParse library
#
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

#
# nlohmann_json library
#
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    )
FetchContent_MakeAvailable(json)

add_executable (${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json argparse)

if(USE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    if(USE_OPENMP_PROFILIZE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDO_PROFILIZE")
    endif()
endif()

if(TRANSFER_BUILD_EXAMPLES)
    set(EXAMPLES_DIR "${CMAKE_SOURCE_DIR}/examples")
    # Create a custom command to copy the file after the build
    add_custom_command(
        TARGET ${PROJECT_NAME}  # Specify the target for which the custom command should run
        POST_BUILD               # Specify that this command should run after the build
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${EXAMPLES_DIR} ${EXECUTABLE_OUTPUT_PATH} 
    )
endif()

# # look here https://github.com/google/googletest/tree/main/googletest
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   # Specify the commit you depend on and update it regularly.
#   URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

# enable_testing()
# Now simply link against gtest or gtest_main as needed. Eg
# add_executable(tree_test tree_test.cpp)
# target_link_libraries(tree_test gtest_main)
# add_test(NAME simple_tree_test COMMAND tree_test)
