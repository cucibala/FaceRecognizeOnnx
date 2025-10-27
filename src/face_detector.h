#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct FaceBox {
    cv::Rect box;
    float score;
    cv::Point2f landmarks[5]; // 5个关键点: 左眼, 右眼, 鼻子, 左嘴角, 右嘴角
};

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();
    
    bool loadModel(const std::string& modelPath);
    std::vector<FaceBox> detect(const cv::Mat& image, float scoreThreshold = 0.5f, float nmsThreshold = 0.4f);
    
private:
    void preprocess(const cv::Mat& image, std::vector<float>& inputData, float& scale);
    std::vector<FaceBox> postprocess(const std::vector<float>& outputs, 
                                      const std::vector<int64_t>& outputShape,
                                      float scale, float scoreThreshold, float nmsThreshold);
    float iou(const cv::Rect& box1, const cv::Rect& box2);
    void nms(std::vector<FaceBox>& boxes, float threshold);
    
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
};

