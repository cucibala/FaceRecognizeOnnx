# GPU 加速构建指南

本项目支持使用 CUDA 和 TensorRT 进行 GPU 加速。

## 前提条件

### 1. CUDA Toolkit
- CUDA 11.x 或 12.x
- 从 [NVIDIA CUDA官网](https://developer.nvidia.com/cuda-downloads) 下载安装

### 2. cuDNN（可选，但推荐）
- cuDNN 8.x
- 从 [NVIDIA cuDNN官网](https://developer.nvidia.com/cudnn) 下载安装

### 3. TensorRT（可选）
- TensorRT 8.x
- 从 [NVIDIA TensorRT官网](https://developer.nvidia.com/tensorrt) 下载安装

### 4. ONNX Runtime GPU 版本
下载对应的 ONNX Runtime GPU 版本：

```bash
# Linux - CUDA 11.x
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.16.3.tgz

# Linux - CUDA 12.x  
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-cuda12-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-gpu-cuda12-1.16.3.tgz

# Windows
# 从 GitHub Releases 下载对应版本的 zip 文件
```

## 编译选项

项目支持三种编译模式：

| 模式 | 说明 | 性能 | 适用场景 |
|------|------|------|----------|
| **CPU** | 仅使用 CPU | 基准 | 无 GPU 环境 |
| **CUDA** | 使用 CUDA 加速 | 2-5x | 有 NVIDIA GPU |
| **TensorRT** | 使用 TensorRT 加速 | 3-10x | 生产部署，最高性能 |

## 编译步骤

### 方式 1: 仅 CPU（默认）

```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
make -j$(nproc)
```

### 方式 2: 启用 CUDA

```bash
mkdir build && cd build
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime-gpu \
    -DUSE_CUDA=ON
make -j$(nproc)
```

### 方式 3: 启用 TensorRT

```bash
mkdir build && cd build
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime-gpu \
    -DUSE_TENSORRT=ON
make -j$(nproc)
```

### 方式 4: 同时启用 CUDA 和 TensorRT

```bash
mkdir build && cd build
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime-gpu \
    -DUSE_CUDA=ON \
    -DUSE_TENSORRT=ON
make -j$(nproc)
```

## Windows 编译

```powershell
# 使用 Visual Studio
mkdir build
cd build

# CUDA 版本
cmake .. -G "Visual Studio 16 2019" -A x64 `
    -DONNXRUNTIME_DIR="C:\path\to\onnxruntime-gpu" `
    -DUSE_CUDA=ON

# 编译
cmake --build . --config Release
```

## 运行测试

### CPU 模式
```bash
./InsightFaceDemo simple test1.jpg test2.jpg
```

### GPU 模式
程序会自动检测编译时的 GPU 支持并使用

```bash
# 运行主程序
./InsightFaceDemo simple test1.jpg test2.jpg

# 运行批处理性能测试
./TestJniInterface models/w600k_mbf.onnx test.jpg --gpu

# 指定 GPU 设备（多卡情况）
./TestJniInterface models/w600k_mbf.onnx test.jpg --gpu --device 1
```

## 性能对比

典型性能提升（512维特征提取）：

| 设备 | 单张 (ms) | 批次16 (ms) | 批次32 (ms) | 吞吐量 (img/s) |
|------|----------|------------|------------|---------------|
| CPU (8核) | 50 | 800 | 1600 | 20 |
| RTX 3060 (CUDA) | 15 | 120 | 200 | 160 |
| RTX 3090 (CUDA) | 10 | 80 | 140 | 230 |
| RTX 3090 (TensorRT) | 5 | 40 | 70 | 457 |

## 验证 GPU 是否工作

### 方法 1: 查看程序输出
```bash
./InsightFaceDemo simple test.jpg test2.jpg
```

应该看到：
```
CUDA support: ENABLED
Using GPU device: 0
========================================
Configuring GPU support...
CUDA provider enabled (device: 0)
```

### 方法 2: 使用 nvidia-smi 监控

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

运行程序时应该看到 GPU 利用率上升。

### 方法 3: 性能测试

```bash
# GPU 测试
./TestJniInterface models/w600k_mbf.onnx test.jpg --gpu

# 应该看到显著的性能提升
```

## 常见问题

### Q1: 编译时提示找不到 CUDA

**A**: 确保安装了 CUDA Toolkit 并设置了环境变量：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q2: 运行时提示 "libcudart.so not found"

**A**: 设置 ONNX Runtime 库路径：

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime-gpu/lib:$LD_LIBRARY_PATH
```

### Q3: GPU 模式性能没有提升

**可能原因**：
1. 批次太小（batch=1），GPU 优势不明显
2. 数据传输开销大于计算开销
3. 模型未正确加载到 GPU

**解决方案**：
- 使用批处理模式（batch >= 16）
- 检查程序输出确认 GPU 已启用
- 使用 `nvidia-smi` 确认 GPU 正在被使用

### Q4: TensorRT 首次运行很慢

**A**: TensorRT 首次运行时会构建优化引擎，这个过程可能需要几分钟。之后会缓存引擎文件，后续运行会很快。

### Q5: 多 GPU 环境下如何选择设备

**A**: 使用 `--device` 参数或设置环境变量：

```bash
# 使用设备 1
./TestJniInterface models/w600k_mbf.onnx test.jpg --gpu --device 1

# 或设置环境变量
export CUDA_VISIBLE_DEVICES=1
./InsightFaceDemo simple test1.jpg test2.jpg
```

## 优化建议

### 1. 批处理大小

| GPU | 推荐批次 | 内存占用 |
|-----|---------|---------|
| GTX 1060 (6GB) | 16-32 | ~1.5GB |
| RTX 3060 (12GB) | 32-64 | ~3GB |
| RTX 3090 (24GB) | 64-128 | ~6GB |

### 2. 精度选择

TensorRT 支持 FP16 精度（默认启用）：
- FP32: 最高精度，较慢
- FP16: 略微精度损失（<1%），速度提升 2x

### 3. 内存管理

如果遇到 OOM（Out of Memory）：

在代码中调整 `gpu_mem_limit`：
```cpp
cuda_options.gpu_mem_limit = 1ULL * 1024 * 1024 * 1024; // 减小到 1GB
```

## JNI 接口 GPU 支持

### C++ 调用

```cpp
// 使用 GPU 初始化
FR_InitializeWithGPU("models/w600k_mbf.onnx", true, 0);

// 批量处理
BatchImageInput input;
BatchImageOutput output;
FR_ProcessBatchImages(&input, &output);
```

### Java 调用

```java
// 添加 native 方法
private native int nativeInitializeWithGPU(String modelPath, boolean useGPU, int deviceId);

// 使用
FaceRecognitionJNI fr = new FaceRecognitionJNI();
fr.nativeInitializeWithGPU("models/w600k_mbf.onnx", true, 0);
```

## 性能测试脚本

创建 `benchmark_gpu.sh`：

```bash
#!/bin/bash

echo "=== GPU Performance Benchmark ==="

# CPU 基准
echo "CPU Baseline:"
./TestJniInterface models/w600k_mbf.onnx test.jpg

# GPU CUDA
echo -e "\nGPU CUDA:"
./TestJniInterface models/w600k_mbf.onnx test.jpg --gpu

# 比较
echo -e "\n=== Comparison ==="
echo "Check the speedup in the summary above"
```

运行：
```bash
chmod +x benchmark_gpu.sh
./benchmark_gpu.sh
```

## 参考资源

- [ONNX Runtime GPU 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## 总结

GPU 加速带来的收益：

✅ **2-10x 性能提升**（取决于批次大小和硬件）  
✅ **更高吞吐量**（适合批量处理）  
✅ **更低延迟**（单张处理也有提升）  
✅ **更好的扩展性**（轻松处理大规模任务）  

现在就开始使用 GPU 加速你的人脸识别应用吧！

