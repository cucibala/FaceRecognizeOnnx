#include "face_detector.h"
#include <algorithm>
#include <iostream>

FaceDetector::FaceDetector(bool useGPU, int deviceId) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "FaceDetector"),
      session_(nullptr),
      inputWidth_(640),
      inputHeight_(640),
      useGPU_(useGPU),
      deviceId_(deviceId) {
    
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // 如果启用 GPU，配置 GPU 选项
    if (useGPU_) {
        setupGPU();
    }
}

FaceDetector::~FaceDetector() {
    if (session_) {
        delete session_;
    }
}

bool FaceDetector::loadModel(const std::string& modelPath) {
    try {
        #ifdef _WIN32
            std::wstring wideModelPath(modelPath.begin(), modelPath.end());
            session_ = new Ort::Session(env_, wideModelPath.c_str(), sessionOptions_);
        #else
            session_ = new Ort::Session(env_, modelPath.c_str(), sessionOptions_);
        #endif
        
        // 获取输入信息
        size_t numInputNodes = session_->GetInputCount();
        if (numInputNodes > 0) {
            Ort::AllocatedStringPtr inputNameAllocated = session_->GetInputNameAllocated(0, allocator_);
            inputNames_.push_back(std::string(inputNameAllocated.get()));
            
            Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
            auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            inputShape_ = tensorInfo.GetShape();
            
            if (inputShape_.size() == 4) {
                // 检查是否为动态维度（-1），如果是则保持默认值
                int64_t modelHeight = inputShape_[2];
                int64_t modelWidth = inputShape_[3];
                
                if (modelHeight > 0) {
                    inputHeight_ = static_cast<int>(modelHeight);
                }
                if (modelWidth > 0) {
                    inputWidth_ = static_cast<int>(modelWidth);
                }
                
                std::cout << "Model input shape: [" << inputShape_[0] << ", " << inputShape_[1] 
                          << ", " << inputShape_[2] << ", " << inputShape_[3] << "]" << std::endl;
                if (modelHeight <= 0 || modelWidth <= 0) {
                    std::cout << "Note: Dynamic dimensions detected (-1), using default size: " 
                              << inputWidth_ << "x" << inputHeight_ << std::endl;
                }
            }
        }
        
        // 获取输出信息
        size_t numOutputNodes = session_->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            Ort::AllocatedStringPtr outputNameAllocated = session_->GetOutputNameAllocated(i, allocator_);
            outputNames_.push_back(std::string(outputNameAllocated.get()));
        }
        
        // 创建指针数组
        inputNamePtrs_.clear();
        for (const auto& name : inputNames_) {
            inputNamePtrs_.push_back(name.c_str());
        }
        
        outputNamePtrs_.clear();
        for (const auto& name : outputNames_) {
            outputNamePtrs_.push_back(name.c_str());
        }
        
        std::cout << "Face detector model loaded successfully!" << std::endl;
        std::cout << "Using input size: " << inputWidth_ << "x" << inputHeight_ << std::endl;
        std::cout << "Number of outputs: " << outputNames_.size() << std::endl;
        for (size_t i = 0; i < outputNames_.size(); i++) {
            std::cout << "  Output " << i << ": " << outputNames_[i] << std::endl;
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading face detector model: " << e.what() << std::endl;
        return false;
    }
}

void FaceDetector::preprocess(const cv::Mat& image, std::vector<float>& inputData, float& scale) {
    // 验证输入图像
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        std::cerr << "Invalid input image for preprocessing" << std::endl;
        scale = 1.0f;
        return;
    }
    
    // 计算缩放比例
    float scaleW = static_cast<float>(inputWidth_) / image.cols;
    float scaleH = static_cast<float>(inputHeight_) / image.rows;
    scale = std::min(scaleW, scaleH);
    
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    
    // 确保新尺寸有效
    if (newWidth <= 0 || newHeight <= 0) {
        std::cerr << "Invalid resize dimensions: " << newWidth << "x" << newHeight << std::endl;
        scale = 1.0f;
        return;
    }
    
    // 调整图像大小
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newWidth, newHeight));
    
    // 创建填充图像
    cv::Mat padded = cv::Mat::zeros(inputHeight_, inputWidth_, CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, newWidth, newHeight)));
    
    // BGR转RGB并归一化
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    
    // 转换为float并归一化到[-1, 1]
    inputData.resize(inputWidth_ * inputHeight_ * 3);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputHeight_; ++h) {
            for (int w = 0; w < inputWidth_; ++w) {
                inputData[c * inputHeight_ * inputWidth_ + h * inputWidth_ + w] = 
                    (rgb.at<cv::Vec3b>(h, w)[c] - 127.5f) / 128.0f;
            }
        }
    }
}

std::vector<FaceBox> FaceDetector::detect(const cv::Mat& image, float scoreThreshold, float nmsThreshold) {
    std::vector<FaceBox> faces;
    
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return faces;
    }
    
    // 验证输入图像
    if (image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return faces;
    }
    
    if (image.cols <= 0 || image.rows <= 0) {
        std::cerr << "Invalid image dimensions: " << image.cols << "x" << image.rows << std::endl;
        return faces;
    }
    
    // 预处理
    std::vector<float> inputData;
    float scale;
    preprocess(image, inputData, scale);
    
    // 检查预处理结果
    if (inputData.empty()) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return faces;
    }
    
    // 创建输入tensor
    std::vector<int64_t> inputShapeBatch = {1, 3, inputHeight_, inputWidth_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(),
        inputShapeBatch.data(), inputShapeBatch.size()
    );
    
    // 推理
    try {
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNamePtrs_.data(), &inputTensor, 1,
            outputNamePtrs_.data(), outputNamePtrs_.size()
        );
        
        // 处理输出
        if (outputTensors.size() == 0) {
            std::cerr << "No output tensors received!" << std::endl;
            return faces;
        }
        
        std::cout << "Number of output tensors: " << outputTensors.size() << std::endl;
        
        // 打印所有输出的形状
        for (size_t i = 0; i < outputTensors.size(); i++) {
            auto shape = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Output " << i << " shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // SCRFD 有多个输出 (scores, bboxes, kps)，需要分别处理
        // 这里使用简化的处理：假设第一个输出包含所有信息
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }
        
        std::vector<float> outputs(outputData, outputData + outputSize);
        faces = postprocess(outputs, outputShape, scale, scoreThreshold, nmsThreshold);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }
    
    return faces;
}

std::vector<FaceBox> FaceDetector::postprocess(const std::vector<float>& outputs,
                                                const std::vector<int64_t>& outputShape,
                                                float scale, float scoreThreshold, float nmsThreshold) {
    std::vector<FaceBox> boxes;
    
    std::cout << "Postprocessing with shape size: " << outputShape.size() << std::endl;
    if (outputShape.size() > 0) {
        std::cout << "Shape dimensions: ";
        for (auto dim : outputShape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
    
    // SCRFD输出格式可能是:
    // 1. [batch, num_anchors, 15] (x1, y1, x2, y2, score, 10 landmarks)
    // 2. [batch, num_anchors, num_classes] 等多种格式
    
    if (outputShape.size() == 3 && outputShape[2] >= 15) {
        // 格式: [batch, num_anchors, 15+]
        int numAnchors = static_cast<int>(outputShape[1]);
        int featDim = static_cast<int>(outputShape[2]);
        
        std::cout << "Processing " << numAnchors << " anchors with " << featDim << " features" << std::endl;
        
        for (int i = 0; i < numAnchors; i++) {
            int offset = i * featDim;
            float score = outputs[offset + 4];
            
            if (score > scoreThreshold) {
                FaceBox face;
                float x1 = outputs[offset + 0] / scale;
                float y1 = outputs[offset + 1] / scale;
                float x2 = outputs[offset + 2] / scale;
                float y2 = outputs[offset + 3] / scale;
                
                face.box = cv::Rect(
                    static_cast<int>(x1),
                    static_cast<int>(y1),
                    static_cast<int>(x2 - x1),
                    static_cast<int>(y2 - y1)
                );
                face.score = score;
                
                // 提取关键点
                if (featDim >= 15) {
                    for (int j = 0; j < 5; j++) {
                        face.landmarks[j].x = outputs[offset + 5 + j * 2] / scale;
                        face.landmarks[j].y = outputs[offset + 5 + j * 2 + 1] / scale;
                    }
                }
                
                boxes.push_back(face);
            }
        }
    } else if (outputShape.size() == 2) {
        // 格式: [num_detections, 15+]
        int numDetections = static_cast<int>(outputShape[0]);
        int featDim = static_cast<int>(outputShape[1]);
        
        std::cout << "Processing " << numDetections << " detections with " << featDim << " features" << std::endl;
        
        for (int i = 0; i < numDetections; i++) {
            int offset = i * featDim;
            
            // 根据特征维度判断格式
            float score;
            float x1, y1, x2, y2;
            
            if (featDim >= 15) {
                // 假设格式: x1, y1, x2, y2, score, kps...
                x1 = outputs[offset + 0] / scale;
                y1 = outputs[offset + 1] / scale;
                x2 = outputs[offset + 2] / scale;
                y2 = outputs[offset + 3] / scale;
                score = outputs[offset + 4];
            } else {
                // 简化格式
                continue;
            }
            
            if (score > scoreThreshold) {
                FaceBox face;
                face.box = cv::Rect(
                    static_cast<int>(x1),
                    static_cast<int>(y1),
                    static_cast<int>(x2 - x1),
                    static_cast<int>(y2 - y1)
                );
                face.score = score;
                
                // 提取关键点
                if (featDim >= 15) {
                    for (int j = 0; j < 5; j++) {
                        face.landmarks[j].x = outputs[offset + 5 + j * 2] / scale;
                        face.landmarks[j].y = outputs[offset + 5 + j * 2 + 1] / scale;
                    }
                }
                
                boxes.push_back(face);
            }
        }
    } else {
        std::cout << "Warning: Unexpected output shape format" << std::endl;
    }
    
    std::cout << "Found " << boxes.size() << " faces before NMS" << std::endl;
    
    // NMS
    nms(boxes, nmsThreshold);
    
    std::cout << "Found " << boxes.size() << " faces after NMS" << std::endl;
    
    return boxes;
}

float FaceDetector::iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    int inter = w * h;
    
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;
    
    return static_cast<float>(inter) / (area1 + area2 - inter);
}

void FaceDetector::setupGPU() {
    try {
        std::cout << "Configuring CUDA GPU support for detector..." << std::endl;
        
#ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = deviceId_;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024; // 2GB
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        
        sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "✓ CUDA provider enabled for detector (GPU device: " << deviceId_ << ")" << std::endl;
#else
        std::cerr << "Warning: GPU requested but not compiled with CUDA support!" << std::endl;
        std::cerr << "Please recompile with: cmake -DUSE_CUDA=ON .." << std::endl;
        std::cerr << "Falling back to CPU execution" << std::endl;
#endif
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error setting up CUDA: " << e.what() << std::endl;
        std::cerr << "Possible reasons:" << std::endl;
        std::cerr << "  - CUDA not installed or not in PATH" << std::endl;
        std::cerr << "  - Using CPU version of ONNX Runtime instead of GPU version" << std::endl;
        std::cerr << "  - Incompatible CUDA version" << std::endl;
        std::cerr << "Falling back to CPU execution" << std::endl;
    }
}

void FaceDetector::nms(std::vector<FaceBox>& boxes, float threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const FaceBox& a, const FaceBox& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            
            float iouVal = iou(boxes[i].box, boxes[j].box);
            if (iouVal > threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    std::vector<FaceBox> result;
    for (size_t i = 0; i < boxes.size(); i++) {
        if (!suppressed[i]) {
            result.push_back(boxes[i]);
        }
    }
    
    boxes = result;
}

