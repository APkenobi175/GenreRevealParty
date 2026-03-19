cmake_minimum_required(VERSION 3.13)
project(GenreRevealPartySerial)

set(SOURCE_FILES main.cpp)
add_executable(GenreRevealPartySerial ${SOURCE_FILES})
target_include_directories(GenreRevealPartySerial PUBLIC "../include")
