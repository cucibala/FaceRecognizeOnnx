cmake_minimum_required(VERSION 3.15)
project(InsightFaceDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU 支持选项
option(USE_CUDA "Enable CUDA support for ONNX Runtime" OFF)
option(USE_TENSORRT "Enable TensorRT support for ONNX Runtime" OFF)

# OpenCV
set(OpenCV_DIR "/home/xxx/codeSource/history/ai-center/3rdparty/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
# 请根据你的ONNX Runtime安装路径修改
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

if(NOT ONNXRUNTIME_DIR)
    message(WARNING "ONNXRUNTIME_DIR not set. Please set it to your ONNX Runtime installation path.")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 配置编译选项
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support enabled")
endif()

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    message(STATUS "TensorRT support enabled")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/face_detector.cpp
    src/face_recognizer.cpp
    src/JniInterface.cpp
)

set(HEADERS
    src/face_detector.h
    src/face_recognizer.h
    src/JniInterface.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    onnxruntime
)

# JNI Test executable
add_executable(TestJniInterface 
    src/test_jni.cpp
    src/face_detector.cpp
    src/face_recognizer.cpp
    src/JniInterface.cpp
)

target_link_libraries(TestJniInterface 
    ${OpenCV_LIBS}
    onnxruntime
)

# Copy models directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

