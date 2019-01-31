
if(CMAKE_TOOLCHAIN_FILE)
set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 2.8.10)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
project(TEEClassifierDemo)

if(WIN32)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-DNOMINMAX)
else()
    add_definitions(-Wall -Wextra -Wno-unused-function)

    add_definitions(-fPIC)
    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        add_definitions(-Ofast)
        add_definitions(-O3)

        add_definitions(-ffast-math)
    endif()
endif()

##############################################
#Set the path to the SAI API
set(API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../api/)
include_directories(${API_PATH}/include)
link_directories(${API_PATH}/lib/linux)

#Set the path to your opencv directory
set(DEP_PATH /Set the path to your opencv directory)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${DEP_PATH}/include)
link_directories(${DEP_PATH}/lib)

file(GLOB demo_src ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/*.h
		    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/*.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Classification/*.h
		    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Classification/*.cpp)

set(inferenceDemo_SRCS
	${demo_src}
)

add_EXECUTABLE(TEEClassifierDemo ${inferenceDemo_SRCS})

target_link_libraries(TEEClassifierDemo TEEClassifier ftd3xx opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc crypto udev pthread)
