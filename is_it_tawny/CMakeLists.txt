cmake_minimum_required(VERSION 3.10)
project(tawny_infer)

find_package(OpenCV REQUIRED)

add_executable(infer infer.cpp)
target_link_libraries(infer ${OpenCV_LIBS})
