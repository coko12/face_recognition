cmake_minimum_required(VERSION 2.8)
project( facenet )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

aux_source_directory(./src/ SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

include_directories( ${CMAKE_CURRENT_LIST_DIR}/../ncnn/build/src )
include_directories( ${CMAKE_CURRENT_LIST_DIR}/../ncnn/src )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( facenet ${OpenCV_LIBS} ${CMAKE_CURRENT_LIST_DIR}/../ncnn/build/src/libncnn.a)
