#include "face_detector.h"
#include <algorithm>
#include <iostream>

FaceDetector::FaceDetector() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "FaceDetector"),
      session_(nullptr),
      inputWidth_(640),
      inputHeight_(640) {
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
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
            inputNames_.push_back(inputNameAllocated.get());
            
            Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
            auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            inputShape_ = tensorInfo.GetShape();
            
            if (inputShape_.size() == 4) {
                inputHeight_ = static_cast<int>(inputShape_[2]);
                inputWidth_ = static_cast<int>(inputShape_[3]);
            }
        }
        
        // 获取输出信息
        size_t numOutputNodes = session_->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            Ort::AllocatedStringPtr outputNameAllocated = session_->GetOutputNameAllocated(i, allocator_);
            outputNames_.push_back(outputNameAllocated.get());
        }
        
        std::cout << "Face detector model loaded successfully!" << std::endl;
        std::cout << "Input shape: [" << inputShape_[0] << ", " << inputShape_[1] 
                  << ", " << inputShape_[2] << ", " << inputShape_[3] << "]" << std::endl;
        
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
            inputNames_.data(), &inputTensor, 1,
            outputNames_.data(), outputNames_.size()
        );
        
        // 处理输出
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
    
    // SCRFD输出格式: [batch, num_anchors, 15] (x1, y1, x2, y2, score, 10 landmarks)
    if (outputShape.size() >= 2) {
        int numAnchors = static_cast<int>(outputShape[1]);
        
        for (int i = 0; i < numAnchors; i++) {
            int offset = i * 15;
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
                for (int j = 0; j < 5; j++) {
                    face.landmarks[j].x = outputs[offset + 5 + j * 2] / scale;
                    face.landmarks[j].y = outputs[offset + 5 + j * 2 + 1] / scale;
                }
                
                boxes.push_back(face);
            }
        }
    }
    
    // NMS
    nms(boxes, nmsThreshold);
    
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

