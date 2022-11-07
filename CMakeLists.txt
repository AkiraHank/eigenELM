cmake_minimum_required(VERSION 3.17)
project(eigenELM)

set(CMAKE_CXX_STANDARD 14)
#set_target_properties(eigenELM PROPERTIES LINKER_LANGUAGE CXX)

aux_source_directory(./src SRC)
include_directories(./include /usr/local/include /Applications/MATLAB_R2022b.app/extern/include ./)

add_executable(eigenELM ${SRC})
