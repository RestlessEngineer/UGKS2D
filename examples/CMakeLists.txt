cmake_minimum_required(VERSION 2.8.12)

project(examples)

# set_target_properties(
#         box PROPERTIES
#         CXX_STANDARD 14
#         CXX_STANDARD_REQUIRED ON
# )

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