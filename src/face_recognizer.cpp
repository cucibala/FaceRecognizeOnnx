#include "face_recognizer.h"
#include <cmath>
#include <iostream>

FaceRecognizer::FaceRecognizer()
    : env_(ORT_LOGGING_LEVEL_WARNING, "FaceRecognizer"),
      session_(nullptr),
      inputWidth_(112),
      inputHeight_(112),
      featureDim_(512) {
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

FaceRecognizer::~FaceRecognizer() {
    if (session_) {
        delete session_;
    }
}

bool FaceRecognizer::loadModel(const std::string& modelPath) {
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
        
        std::cout << "Face recognizer model loaded successfully!" << std::endl;
        std::cout << "Using input size: " << inputWidth_ << "x" << inputHeight_ << std::endl;
        std::cout << "Number of outputs: " << outputNames_.size() << std::endl;
        for (size_t i = 0; i < outputNames_.size(); i++) {
            std::cout << "  Output " << i << ": " << outputNames_[i] << std::endl;
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading face recognizer model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat FaceRecognizer::alignFace(const cv::Mat& image, const FaceBox& face) {
    // 验证输入
    if (image.empty()) {
        std::cerr << "Empty image in alignFace" << std::endl;
        return cv::Mat();
    }
    
    // 标准5点模板（112x112图像）
    cv::Point2f dstLandmarks[5] = {
        cv::Point2f(38.2946f, 51.6963f),  // 左眼
        cv::Point2f(73.5318f, 51.5014f),  // 右眼
        cv::Point2f(56.0252f, 71.7366f),  // 鼻子
        cv::Point2f(41.5493f, 92.3655f),  // 左嘴角
        cv::Point2f(70.7299f, 92.2041f)   // 右嘴角
    };
    
    // 使用相似变换对齐人脸
    cv::Mat M = cv::estimateAffinePartial2D(
        std::vector<cv::Point2f>(face.landmarks, face.landmarks + 5),
        std::vector<cv::Point2f>(dstLandmarks, dstLandmarks + 5)
    );
    
    // 检查变换矩阵是否有效
    if (M.empty()) {
        std::cerr << "Failed to compute affine transformation" << std::endl;
        // 如果无法计算变换，直接裁剪并缩放人脸区域
        cv::Rect safeFace = face.box & cv::Rect(0, 0, image.cols, image.rows);
        if (safeFace.width > 0 && safeFace.height > 0) {
            cv::Mat cropped = image(safeFace);
            cv::Mat resized;
            cv::resize(cropped, resized, cv::Size(inputWidth_, inputHeight_));
            return resized;
        }
        return cv::Mat();
    }
    
    cv::Mat aligned;
    cv::warpAffine(image, aligned, M, cv::Size(inputWidth_, inputHeight_));
    
    return aligned;
}

void FaceRecognizer::preprocess(const cv::Mat& aligned, std::vector<float>& inputData) {
    // BGR转RGB
    cv::Mat rgb;
    cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
    
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

std::vector<float> FaceRecognizer::extractFeatureSimple(const cv::Mat& image) {
    std::vector<float> feature;
    
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return feature;
    }
    
    // 验证输入
    if (image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return feature;
    }
    
    std::cout << "Input image size: " << image.cols << "x" << image.rows << std::endl;
    
    // 直接 resize 到模型输入尺寸
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputWidth_, inputHeight_));
    std::cout << "Resized to: " << resized.cols << "x" << resized.rows << std::endl;
    
    // 预处理
    std::vector<float> inputData;
    preprocess(resized, inputData);
    
    // 检查预处理结果
    if (inputData.empty()) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return feature;
    }
    
    std::cout << "Preprocessed data size: " << inputData.size() << std::endl;
    
    // 创建输入tensor
    std::vector<int64_t> inputShapeBatch = {1, 3, inputHeight_, inputWidth_};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(),
        inputShapeBatch.data(), inputShapeBatch.size()
    );
    
    // 推理
    try {
        std::cout << "Running inference..." << std::endl;
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNamePtrs_.data(), &inputTensor, 1,
            outputNamePtrs_.data(), outputNamePtrs_.size()
        );
        
        std::cout << "Inference completed, processing outputs..." << std::endl;
        
        // 获取特征
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "Output shape: [";
        for (size_t i = 0; i < outputShape.size(); i++) {
            std::cout << outputShape[i];
            if (i < outputShape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        size_t featureSize = 1;
        for (auto dim : outputShape) {
            featureSize *= dim;
        }
        
        std::cout << "Feature size: " << featureSize << std::endl;
        
        feature.assign(outputData, outputData + featureSize);
        
        // L2归一化
        normalize(feature);
        
        std::cout << "Feature extraction successful!" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during feature extraction: " << e.what() << std::endl;
    }
    
    return feature;
}

std::vector<float> FaceRecognizer::extractFeature(const cv::Mat& image, const FaceBox& face) {
    std::vector<float> feature;
    
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return feature;
    }
    
    // 验证输入
    if (image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return feature;
    }
    
    // 对齐人脸
    cv::Mat aligned = alignFace(image, face);
    
    // 检查对齐结果
    if (aligned.empty()) {
        std::cerr << "Face alignment failed!" << std::endl;
        return feature;
    }
    
    // 预处理
    std::vector<float> inputData;
    preprocess(aligned, inputData);
    
    // 检查预处理结果
    if (inputData.empty()) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return feature;
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
        
        // 获取特征
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t featureSize = 1;
        for (auto dim : outputShape) {
            featureSize *= dim;
        }
        
        feature.assign(outputData, outputData + featureSize);
        
        // L2归一化
        normalize(feature);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during feature extraction: " << e.what() << std::endl;
    }
    
    return feature;
}

void FaceRecognizer::normalize(std::vector<float>& feature) {
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0) {
        for (float& val : feature) {
            val /= norm;
        }
    }
}

float FaceRecognizer::compareFaces(const std::vector<float>& feature1, const std::vector<float>& feature2) {
    if (feature1.size() != feature2.size() || feature1.empty()) {
        return 0.0f;
    }
    
    // 计算余弦相似度
    float dotProduct = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        dotProduct += feature1[i] * feature2[i];
    }
    
    // 由于特征已经归一化，余弦相似度就是点积
    // 将[-1, 1]映射到[0, 1]
    return (dotProduct + 1.0f) / 2.0f;
}

