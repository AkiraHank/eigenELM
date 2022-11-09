cmake_minimum_required(VERSION 3.17)
project(eigenELM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
#set_target_properties(eigenELM PROPERTIES LINKER_LANGUAGE CXX)

aux_source_directory(./src SRC)
include_directories(./include ./)

add_executable(eigenELM ${SRC})
