#include "face_detector.h"
#include <algorithm>
#include <iostream>

FaceDetector::FaceDetector(bool useGPU, int deviceId) 
    : env_(ORT_LOGGING_LEVEL_ERROR, "FaceDetector"),  // 改为 ERROR 级别，忽略警告
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
    std::vector<FaceBox> allBoxes;
    
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return allBoxes;
    }
    
    // 验证输入图像
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        std::cerr << "Invalid input image!" << std::endl;
        return allBoxes;
    }
    
    // 预处理
    std::vector<float> inputData;
    float det_scale;
    preprocess(image, inputData, det_scale);
    
    if (inputData.empty()) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return allBoxes;
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
        
        if (outputTensors.size() == 0) {
            std::cerr << "No output tensors!" << std::endl;
            return allBoxes;
        }
        
        // 判断模型配置
        size_t num_outputs = outputTensors.size();
        int fmc = 0;  // Feature map count
        bool use_kps = false;
        std::vector<int> feat_stride_fpn;
        
        if (num_outputs == 6) {
            fmc = 3;
            feat_stride_fpn = {8, 16, 32};
        } else if (num_outputs == 9) {
            fmc = 3;
            feat_stride_fpn = {8, 16, 32};
            use_kps = true;
        } else if (num_outputs == 10) {
            fmc = 5;
            feat_stride_fpn = {8, 16, 32, 64, 128};
        } else if (num_outputs == 15) {
            fmc = 5;
            feat_stride_fpn = {8, 16, 32, 64, 128};
            use_kps = true;
        } else {
            std::cerr << "Unsupported number of outputs: " << num_outputs << std::endl;
            return allBoxes;
        }
        
        std::cout << "SCRFD outputs: " << num_outputs << ", fmc=" << fmc << ", use_kps=" << use_kps << std::endl;
        
        // 处理每个 scale 的输出
        for (int idx = 0; idx < fmc; idx++) {
            int stride = feat_stride_fpn[idx];
            
            // 获取 scores, bbox_preds, kps_preds
            auto scores_tensor = outputTensors[idx];
            auto bbox_preds_tensor = outputTensors[idx + fmc];
            
            auto scores_shape = scores_tensor.GetTensorTypeAndShapeInfo().GetShape();
            float* scores_data = scores_tensor.GetTensorMutableData<float>();
            float* bbox_preds_data = bbox_preds_tensor.GetTensorMutableData<float>();
            
            float* kps_preds_data = nullptr;
            if (use_kps) {
                kps_preds_data = outputTensors[idx + fmc * 2].GetTensorMutableData<float>();
            }
            
            // 计算特征图尺寸
            int height = inputHeight_ / stride;
            int width = inputWidth_ / stride;
            int num_anchors = height * width;
            
            // 生成 anchor centers
            std::vector<cv::Point2f> anchor_centers;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    anchor_centers.push_back(cv::Point2f(w * stride, h * stride));
                }
            }
            
            // 处理每个 anchor
            for (int i = 0; i < num_anchors; i++) {
                float score = scores_data[i];
                
                if (score >= scoreThreshold) {
                    FaceBox face;
                    
                    // 解码边界框
                    float bbox_pred[4] = {
                        bbox_preds_data[i * 4] * stride,
                        bbox_preds_data[i * 4 + 1] * stride,
                        bbox_preds_data[i * 4 + 2] * stride,
                        bbox_preds_data[i * 4 + 3] * stride
                    };
                    face.box = distance2bbox(anchor_centers[i], bbox_pred, det_scale);
                    face.score = score;
                    
                    // 解码关键点
                    if (use_kps && kps_preds_data) {
                        float kps_pred[10];
                        for (int k = 0; k < 10; k++) {
                            kps_pred[k] = kps_preds_data[i * 10 + k] * stride;
                        }
                        distance2kps(anchor_centers[i], kps_pred, face.landmarks, 5, det_scale);
                    }
                    
                    allBoxes.push_back(face);
                }
            }
        }
        
        std::cout << "Found " << allBoxes.size() << " faces before NMS" << std::endl;
        
        // NMS
        nms(allBoxes, nmsThreshold);
        
        std::cout << "Found " << allBoxes.size() << " faces after NMS" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }
    
    return allBoxes;
}

// SCRFD 核心函数：distance2bbox（从 distance 解码为 bbox）
cv::Rect distance2bbox(const cv::Point2f& anchor_center, const float* distance, float scale) {
    float x1 = (anchor_center.x - distance[0]) / scale;
    float y1 = (anchor_center.y - distance[1]) / scale;
    float x2 = (anchor_center.x + distance[2]) / scale;
    float y2 = (anchor_center.y + distance[3]) / scale;
    return cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
}

// SCRFD 核心函数：distance2kps（从 distance 解码为关键点）
void distance2kps(const cv::Point2f& anchor_center, const float* distance, 
                  cv::Point2f* kps, int num_kps, float scale) {
    for (int i = 0; i < num_kps; i++) {
        kps[i].x = (anchor_center.x + distance[i * 2]) / scale;
        kps[i].y = (anchor_center.y + distance[i * 2 + 1]) / scale;
    }
}

std::vector<FaceBox> FaceDetector::postprocess(const std::vector<float>& outputs,
                                                const std::vector<int64_t>& outputShape,
                                                float scale, float scoreThreshold, float nmsThreshold) {
    std::vector<FaceBox> boxes;
    
    // ！！！这是错误的实现 - SCRFD 模型有多个输出头
    // 你需要在 detect() 函数中正确处理所有输出
    std::cerr << "ERROR: postprocess should not be called with single output!" << std::endl;
    std::cerr << "SCRFD has multiple outputs (scores, bbox_preds, kps_preds for each scale)" << std::endl;
    
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
        std::cout << "Configuring GPU support for detector..." << std::endl;
        
#ifdef USE_TENSORRT
        // TensorRT - 尝试添加，失败则跳过
        try {
            std::cout << "  Attempting to add TensorRT provider for detector..." << std::endl;
            
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = deviceId_;
            trt_options.trt_engine_cache_enable = 1;
            trt_options.trt_engine_cache_path = "./trt_cache";
            trt_options.trt_max_workspace_size = 2ULL * 1024 * 1024 * 1024;
            trt_options.trt_fp16_enable = 1;
            trt_options.trt_int8_enable = 0;
            trt_options.trt_dla_enable = 0;
            trt_options.trt_dump_subgraphs = 0;
            
            sessionOptions_.AppendExecutionProvider_TensorRT(trt_options);
            std::cout << "✓ TensorRT provider added for detector" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "⚠️  TensorRT failed for detector: " << e.what() << std::endl;
            std::cerr << "   Will use CUDA instead" << std::endl;
        } catch (...) {
            std::cerr << "⚠️  TensorRT failed for detector (unknown error)" << std::endl;
            std::cerr << "   Will use CUDA instead" << std::endl;
        }
#endif

#ifdef USE_CUDA
        // CUDA - 更稳定的选择
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = deviceId_;
            
            sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "✓ CUDA provider added for detector (device: " << deviceId_ << ")" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "⚠️  Failed to add CUDA provider: " << e.what() << std::endl;
            std::cerr << "   Falling back to CPU" << std::endl;
        }
#endif

#if !defined(USE_CUDA) && !defined(USE_TENSORRT)
        std::cerr << "Warning: GPU requested but not compiled with CUDA/TensorRT support!" << std::endl;
        std::cerr << "Using CPU execution" << std::endl;
#endif
        
    } catch (const std::exception& e) {
        std::cerr << "Error setting up GPU: " << e.what() << std::endl;
        std::cerr << "Using CPU execution" << std::endl;
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

