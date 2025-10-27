#include "face_recognizer.h"
#include <cmath>
#include <iostream>

FaceRecognizer::FaceRecognizer(bool useGPU, int deviceId)
    : env_(ORT_LOGGING_LEVEL_ERROR, "FaceRecognizer"),  // 改为 ERROR 级别，忽略警告
      session_(nullptr),
      inputWidth_(112),
      inputHeight_(112),
      featureDim_(512),
      useGPU_(useGPU),
      deviceId_(deviceId) {
    
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // 如果启用 GPU，配置 GPU 选项
    if (useGPU_) {
        setupGPU();
    }
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
    
    // 直接 resize 到模型输入尺寸
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputWidth_, inputHeight_));
    
    // 预处理
    std::vector<float> inputData;
    preprocess(resized, inputData);
    
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

std::vector<std::vector<float>> FaceRecognizer::extractFeaturesBatchSimple(const std::vector<cv::Mat>& images) {
    std::vector<std::vector<float>> features;
    
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return features;
    }
    
    if (images.empty()) {
        std::cerr << "No images to process!" << std::endl;
        return features;
    }
    
    int batchSize = images.size();
    try {
        // 优化：一次性分配所有内存
        const int singleImageSize = 3 * inputHeight_ * inputWidth_;
        std::vector<float> batchInputData(batchSize * singleImageSize);
        std::vector<bool> validFlags(batchSize, false);
        
        // 优化：使用 OpenMP 并行处理（如果可用）
        #pragma omp parallel for
        for (int i = 0; i < batchSize; i++) {
            int offset = i * singleImageSize;
            
            if (images[i].empty()) {
                continue;
            }

            // Resize
            cv::Mat resized;
            cv::resize(images[i], resized, cv::Size(inputWidth_, inputHeight_));
            
            // BGR转RGB并归一化（优化版本）
            cv::Mat rgb;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            
            // 直接写入批次数据（避免额外的 vector 复制）
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < inputHeight_; ++h) {
                    for (int w = 0; w < inputWidth_; ++w) {
                        int idx = offset + c * inputHeight_ * inputWidth_ + h * inputWidth_ + w;
                        batchInputData[idx] = (rgb.at<cv::Vec3b>(h, w)[c] - 127.5f) / 128.0f;
                    }
                }
            }
            
            validFlags[i] = true;
        }
        
        // 创建批量输入tensor
        // ONNX Runtime 会自动处理 CPU->GPU 数据传输
        std::vector<int64_t> inputShapeBatch = {static_cast<int64_t>(batchSize), 3, inputHeight_, inputWidth_};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, batchInputData.data(), batchInputData.size(),
            inputShapeBatch.data(), inputShapeBatch.size()
        );
        
        // 批量推理
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNamePtrs_.data(), &inputTensor, 1,
            outputNamePtrs_.data(), outputNamePtrs_.size()
        );
        
        // 获取输出
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // 解析每个图片的特征
        if (outputShape.size() >= 2) {
            int outputBatchSize = static_cast<int>(outputShape[0]);
            int featureDim = static_cast<int>(outputShape[1]);
            
            for (int i = 0; i < outputBatchSize && i < batchSize; i++) {
                std::vector<float> feature;
                
                if (validFlags[i]) {
                    int offset = i * featureDim;
                    feature.assign(outputData + offset, outputData + offset + featureDim);
                    normalize(feature);
                }
                
                features.push_back(feature);
            }
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during batch feature extraction: " << e.what() << std::endl;
    }
    
    return features;
}

std::vector<std::vector<float>> FaceRecognizer::extractFeaturesBatch(const std::vector<cv::Mat>& images) {
    std::vector<std::vector<float>> features;
    
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return features;
    }
    
    if (images.empty()) {
        std::cerr << "No images to process!" << std::endl;
        return features;
    }
    
    // 注意：由于每张图片可能需要不同的对齐参数，
    // 这里使用逐个处理的方式（未来可以优化为真正的批处理）
    for (size_t i = 0; i < images.size(); i++) {
        std::vector<float> feature = extractFeatureSimple(images[i]);
        features.push_back(feature);
    }
    
    return features;
}

void FaceRecognizer::setupGPU() {
    try {
        std::cout << "Configuring GPU support..." << std::endl;
        
// #ifdef USE_TENSORRT
//         // TensorRT 优先（性能最好）
//         OrtTensorRTProviderOptions trt_options;
//         trt_options.device_id = deviceId_;
//         trt_options.trt_fp16_enable = 1; // 启用 FP16 加速
        
//         sessionOptions_.AppendExecutionProvider_TensorRT(trt_options);
//         std::cout << "✓ TensorRT provider enabled (GPU device: " << deviceId_ << ", FP16: ON)" << std::endl;
// #endif

#ifdef USE_CUDA
        // CUDA 作为备选或独立使用
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = deviceId_;
        
        sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "✓ CUDA provider enabled (GPU device: " << deviceId_ << ")" << std::endl;
#endif

#if !defined(USE_CUDA) && !defined(USE_TENSORRT)
        std::cerr << "Warning: GPU requested but not compiled with CUDA/TensorRT support!" << std::endl;
        std::cerr << "Please recompile with: cmake -DUSE_CUDA=ON or -DUSE_TENSORRT=ON" << std::endl;
        std::cerr << "Falling back to CPU execution" << std::endl;
#endif
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error setting up GPU: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU execution" << std::endl;
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

