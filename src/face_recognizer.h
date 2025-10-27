#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include "face_detector.h"

class FaceRecognizer {
public:
    FaceRecognizer();
    ~FaceRecognizer();
    
    bool loadModel(const std::string& modelPath);
    std::vector<float> extractFeature(const cv::Mat& image, const FaceBox& face);
    std::vector<float> extractFeatureSimple(const cv::Mat& image);  // 直接处理整张图
    
    // 批量处理接口
    std::vector<std::vector<float>> extractFeaturesBatch(const std::vector<cv::Mat>& images);
    std::vector<std::vector<float>> extractFeaturesBatchSimple(const std::vector<cv::Mat>& images);
    
    float compareFaces(const std::vector<float>& feature1, const std::vector<float>& feature2);
    
private:
    cv::Mat alignFace(const cv::Mat& image, const FaceBox& face);
    void preprocess(const cv::Mat& aligned, std::vector<float>& inputData);
    void normalize(std::vector<float>& feature);
    
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions sessionOptions_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;
    std::vector<int64_t> inputShape_;
    
    int inputWidth_;
    int inputHeight_;
    int featureDim_;
};

