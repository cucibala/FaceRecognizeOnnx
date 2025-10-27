#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include "face_detector.h"

class FaceRecognizer {
public:
    FaceRecognizer(bool useGPU = false, int deviceId = 0);
    ~FaceRecognizer();
    
    bool loadModel(const std::string& modelPath);
    std::vector<float> extractFeature(const cv::Mat& image, const FaceBox& face);
    std::vector<float> extractFeatureSimple(const cv::Mat& image);  // 直接处理整张图
    
    // 批量处理接口
    std::vector<std::vector<float>> extractFeaturesBatch(const std::vector<cv::Mat>& images);
    std::vector<std::vector<float>> extractFeaturesBatchSimple(const std::vector<cv::Mat>& images);
    
    // TensorRT 引擎预热（提前构建常用 batch size 的引擎）
    void warmupTensorRT(const std::vector<int>& batchSizes = {1, 16, 32});
    
    float compareFaces(const std::vector<float>& feature1, const std::vector<float>& feature2);
    
private:
    cv::Mat alignFace(const cv::Mat& image, const FaceBox& face);
    void preprocess(const cv::Mat& aligned, std::vector<float>& inputData);
    void normalize(std::vector<float>& feature);
    void setupGPU();
    
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions sessionOptions_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    bool useGPU_;
    int deviceId_;
    
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;
    std::vector<int64_t> inputShape_;
    
    int inputWidth_;
    int inputHeight_;
    int featureDim_;
};

