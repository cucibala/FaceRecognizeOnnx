cmake_minimum_required(VERSION 3.15)
project(InsightFaceDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
# 请根据你的ONNX Runtime安装路径修改
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime directory")

if(NOT ONNXRUNTIME_DIR)
    message(WARNING "ONNXRUNTIME_DIR not set. Please set it to your ONNX Runtime installation path.")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Source files
set(SOURCES
    src/main.cpp
    src/face_detector.cpp
    src/face_recognizer.cpp
)

set(HEADERS
    src/face_detector.h
    src/face_recognizer.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    onnxruntime
)

# Copy models directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

