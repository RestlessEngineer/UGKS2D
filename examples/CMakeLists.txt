cmake_minimum_required(VERSION 3.10)


option(DO_PROFILIZE "profile time steps" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
if(DO_PROFILIZE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDO_PROFILIZE")
endif()

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message("OpenMP library doesn't exist")
endif()


set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(SOURCE  ${SRC_DIR}/tools.cpp 
            ${SRC_DIR}/solver.cpp)

set(HEADER  ${SRC_DIR}/global.h
	    ${SRC_DIR}/tools.h
            ${SRC_DIR}/solver.h)


add_executable(ugks_box box.cpp ${SOURCE})
add_executable(ugks_nozzle simple_supersonic_nozzle.cpp ${SOURCE})
add_executable(ugks_nozzle_start_from_results noozle_from_last_result.cpp ${SOURCE})

target_include_directories(
        ugks_box 
        PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
target_include_directories(
        ugks_nozzle 
        PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_include_directories(
        ugks_nozzle_start_from_results 
        PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)