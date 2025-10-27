# InsightFace C++ Demo - buffalo_sc 模型

基于 InsightFace 的 C++ 人脸检测和识别项目，使用 buffalo_sc 模型进行人脸检测、特征提取和人脸比对。

## 功能特性

- ✅ **人脸检测**: 使用 SCRFD 检测模型 (det_10g.onnx)
- ✅ **关键点检测**: 检测5个人脸关键点（双眼、鼻子、嘴角）
- ✅ **特征提取**: 使用 ArcFace 模型提取512维人脸特征 (w600k_r50.onnx)
- ✅ **人脸比对**: 基于余弦相似度的人脸比对
- ✅ **人脸对齐**: 基于关键点的人脸对齐
- ✅ **实时检测**: 支持摄像头实时人脸检测和识别

## 依赖项

### 必需依赖

1. **OpenCV** (>= 4.x)
   - 用于图像处理和显示

2. **ONNX Runtime** (>= 1.12.0)
   - 用于运行 ONNX 模型推理

3. **CMake** (>= 3.15)
   - 构建工具

### 安装依赖

#### Windows

1. **安装 OpenCV**
   ```bash
   # 使用 vcpkg
   vcpkg install opencv4:x64-windows
   
   # 或者从官网下载预编译版本
   # https://opencv.org/releases/
   ```

2. **安装 ONNX Runtime**
   ```bash
   # 从官网下载预编译版本
   # https://github.com/microsoft/onnxruntime/releases
   
   # 下载 onnxruntime-win-x64-*.zip
   # 解压到某个目录，例如: C:/onnxruntime
   ```

#### Linux

```bash
# 安装 OpenCV
sudo apt-get update
sudo apt-get install libopencv-dev

# 下载 ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
```

#### macOS

```bash
# 安装 OpenCV
brew install opencv

# 下载 ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
tar -xzf onnxruntime-osx-arm64-1.16.3.tgz
```

## 模型文件

本项目使用 InsightFace 的 buffalo_sc 模型，需要以下两个模型文件：

1. **det_10g.onnx** - SCRFD 人脸检测模型
2. **w600k_r50.onnx** - ArcFace 人脸识别模型

### 模型文件结构

```
onnxFRTest/
├── models/
│   ├── det_10g.onnx
│   └── w600k_r50.onnx
├── src/
│   ├── main.cpp
│   ├── face_detector.h
│   ├── face_detector.cpp
│   ├── face_recognizer.h
│   └── face_recognizer.cpp
└── CMakeLists.txt
```

### 获取模型

如果你已经有 buffalo_sc 模型文件，请将它们放在 `models/` 目录下。

## 编译项目

### Windows

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake (需要指定 ONNX Runtime 路径)
cmake .. -DONNXRUNTIME_DIR="C:/path/to/onnxruntime" -DOpenCV_DIR="C:/path/to/opencv/build"

# 编译
cmake --build . --config Release
```

### Linux / macOS

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime

# 编译
make -j4
```

## 使用方法

编译完成后，可执行文件位于 `build/` 目录下。

### 1. 人脸检测

检测图像中的所有人脸并标注关键点：

```bash
./InsightFaceDemo detect path/to/image.jpg
```

### 2. 人脸比对

比对两张图像中的人脸是否为同一人：

```bash
./InsightFaceDemo compare path/to/image1.jpg path/to/image2.jpg
```

输出示例：
```
检测到人脸...
提取特征...
特征维度: 512
相似度: 0.876543
结果: 同一人 (相似度: 0.876543 > 0.6)
```

### 3. 实时检测（摄像头）

使用摄像头进行实时人脸检测和识别：

```bash
./InsightFaceDemo webcam
```

操作说明：
- 按 `s` 键保存当前检测到的第一个人脸作为参考
- 保存后会自动比对摄像头中的人脸与参考人脸
- 按 `q` 键退出

## 项目结构

```
src/
├── face_detector.h/cpp    # 人脸检测器类
│   ├── 加载 SCRFD 检测模型
│   ├── 预处理和后处理
│   ├── NMS 非极大值抑制
│   └── 关键点检测
│
├── face_recognizer.h/cpp  # 人脸识别器类
│   ├── 加载 ArcFace 识别模型
│   ├── 人脸对齐
│   ├── 特征提取
│   └── 人脸比对（余弦相似度）
│
└── main.cpp               # 主程序
    ├── 命令行参数解析
    ├── 检测模式
    ├── 比对模式
    └── 实时检测模式
```

## 技术细节

### 人脸检测流程

1. **预处理**: 图像缩放、填充、BGR→RGB、归一化到[-1, 1]
2. **推理**: ONNX Runtime 推理
3. **后处理**: 解析检测结果、NMS 过滤
4. **输出**: 人脸框、置信度、5个关键点

### 人脸识别流程

1. **人脸对齐**: 基于5个关键点进行仿射变换对齐到112x112
2. **预处理**: BGR→RGB、归一化到[-1, 1]
3. **推理**: ONNX Runtime 推理
4. **特征提取**: 获取512维特征向量
5. **L2归一化**: 特征向量归一化
6. **比对**: 计算余弦相似度

### 相似度阈值

默认阈值为 **0.6**，可以根据实际需求调整：

- **> 0.6**: 认为是同一人
- **≤ 0.6**: 认为是不同人

建议阈值范围：
- 严格模式: 0.7 - 0.8
- 正常模式: 0.5 - 0.6
- 宽松模式: 0.4 - 0.5

## 性能优化

### 多线程推理

在 `face_detector.cpp` 和 `face_recognizer.cpp` 中已启用：

```cpp
sessionOptions_.SetIntraOpNumThreads(4);
sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
```

### GPU 加速

如需使用 GPU 加速（需要 CUDA 版本的 ONNX Runtime）：

```cpp
// 在构造函数中添加
OrtCUDAProviderOptions cuda_options;
sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
```

## 常见问题

### Q1: 编译时找不到 ONNX Runtime

**A**: 确保正确设置 `DONNXRUNTIME_DIR`：

```bash
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
```

### Q2: 运行时提示找不到模型文件

**A**: 确保 `models/` 目录在可执行文件同级目录下，或使用绝对路径。

### Q3: 检测不到人脸

**A**: 尝试：
- 降低检测阈值：`detector.detect(image, 0.3f)`
- 检查图像是否正常加载
- 确保图像中人脸清晰且正面

### Q4: 相似度总是很低

**A**: 检查：
- 人脸是否被正确检测和对齐
- 关键点是否准确
- 模型文件是否完整

## 许可证

本项目仅用于学习和研究目的。

## 参考

- [InsightFace](https://github.com/deepinsight/insightface)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [OpenCV](https://opencv.org/)

## 更新日志

### v1.0.0
- ✅ 初始版本
- ✅ 支持人脸检测
- ✅ 支持人脸识别
- ✅ 支持人脸比对
- ✅ 支持实时检测

## 联系方式

如有问题或建议，欢迎提交 Issue。

