# 模型文件目录

## 说明

请将 InsightFace buffalo_sc 模型文件放置在此目录下。

## 需要的模型文件

### 1. det_10g.onnx
- **类型**: 人脸检测模型 (SCRFD)
- **用途**: 检测图像中的人脸并提取5个关键点
- **输入**: [1, 3, 640, 640] RGB图像，归一化到[-1, 1]
- **输出**: 人脸框、置信度、关键点坐标

### 2. w600k_r50.onnx
- **类型**: 人脸识别模型 (ArcFace)
- **用途**: 提取人脸特征向量
- **输入**: [1, 3, 112, 112] 对齐后的人脸图像，归一化到[-1, 1]
- **输出**: [1, 512] 特征向量

## 模型来源

这些模型来自 InsightFace 的 buffalo_sc 模型包。

如果你从 InsightFace 官方仓库或模型库下载了 buffalo_sc 模型包，
模型文件通常位于：
```
buffalo_sc/
├── det_10g.onnx
└── w600k_r50.onnx
```

## 文件结构

确保此目录下包含以下文件：

```
models/
├── README.md          (本文件)
├── det_10g.onnx       (人脸检测模型)
└── w600k_r50.onnx     (人脸识别模型)
```

## 注意事项

⚠️ 模型文件较大（det_10g.onnx ~17MB, w600k_r50.onnx ~166MB），
   建议不要提交到 Git 仓库。

✅ 确保模型文件的完整性，可以通过文件大小验证：
   - det_10g.onnx: 约 16-17 MB
   - w600k_r50.onnx: 约 166 MB

## 验证

放置模型文件后，运行程序时应该看到：

```
Face detector model loaded successfully!
Input shape: [1, 3, 640, 640]
Face recognizer model loaded successfully!
Input shape: [1, 3, 112, 112]

所有模型加载成功!
```

如果出现加载错误，请检查：
1. 文件路径是否正确
2. 文件是否完整（未损坏）
3. 文件名是否匹配

