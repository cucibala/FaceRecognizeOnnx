cmake_minimum_required(VERSION 3.15)
project(InsightFaceDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译器优化选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # 启用 SSE2/AVX 优化（base64 解码加速）
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -O3")
    message(STATUS "Enabled SSE2 optimizations")
elseif(MSVC)
    # Windows MSVC 默认支持 SSE2
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:SSE2")
    message(STATUS "Enabled SSE2 optimizations (MSVC)")
endif()

# OpenMP 支持（并行图片解码）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP enabled for parallel image decoding")
else()
    message(WARNING "OpenMP not found - image decoding will be sequential")
endif()

# GPU 支持选项
option(USE_CUDA "Enable CUDA support for ONNX Runtime" ON)  # 默认启用 CUDA
option(USE_TENSORRT "Enable TensorRT support for ONNX Runtime" OFF)  # 默认禁用 TensorRT

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
# 请根据你的ONNX Runtime安装路径修改
# set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

if(NOT ONNXRUNTIME_DIR)
    message(WARNING "ONNXRUNTIME_DIR not set. Please set it to your ONNX Runtime installation path.")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 配置 GPU 支持
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support enabled")
endif()

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    message(STATUS "TensorRT support enabled")
endif()

if(USE_CUDA OR USE_TENSORRT)
    message(STATUS "Make sure you are using ONNX Runtime GPU version")
else()
    message(STATUS "Using CPU only (use -DUSE_CUDA=ON or -DUSE_TENSORRT=ON to enable GPU)")
endif()

# Core library sources (不包含 main.cpp 和 test_jni.cpp)
set(LIB_SOURCES
    src/face_detector.cpp
    src/face_recognizer.cpp
    src/JniInterface.cpp
)

set(HEADERS
    src/face_detector.h
    src/face_recognizer.h
    src/JniInterface.h
)

# 创建动态链接库 (共享库)
add_library(image_processor SHARED ${LIB_SOURCES} ${HEADERS})

target_link_libraries(image_processor 
    ${OpenCV_LIBS}
    onnxruntime
)

# 链接 OpenMP（如果可用）
if(OpenMP_CXX_FOUND)
    target_link_libraries(image_processor OpenMP::OpenMP_CXX)
endif()

# 设置库的输出名称和版本
set_target_properties(image_processor PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER "src/JniInterface.h"
)

# 主程序可执行文件
add_executable(${PROJECT_NAME} 
    src/main.cpp
)

# 主程序链接到共享库
target_link_libraries(${PROJECT_NAME} 
    image_processor
    ${OpenCV_LIBS}
    onnxruntime
)

# JNI Test executable
add_executable(TestJniInterface 
    src/test_jni.cpp
)

# 测试程序也链接到共享库
target_link_libraries(TestJniInterface 
    image_processor
    ${OpenCV_LIBS}
    onnxruntime
)

# 测试 .so 库的小程序
add_executable(TestLibrary test_so.cpp)
target_link_libraries(TestLibrary image_processor)

# 安装规则（可选）
install(TARGETS image_processor
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

# Copy models directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

