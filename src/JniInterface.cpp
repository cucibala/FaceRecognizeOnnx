#include "JniInterface.h"
#include "face_recognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

// 全局变量：单例模式的识别器
static std::unique_ptr<FaceRecognizer> g_recognizer = nullptr;
static std::string g_modelPath = "";
static bool g_initialized = false;

// Base64 解码函数
static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::vector<unsigned char> base64_decode(const std::string& encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<unsigned char> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
    }

    return ret;
}

// 将 Base64 字符串转换为 OpenCV Mat
cv::Mat base64ToMat(const char* base64_str, int str_len) {
    try {
        std::string encoded_string(base64_str, str_len);
        std::vector<unsigned char> decoded_data = base64_decode(encoded_string);
        
        if (decoded_data.empty()) {
            std::cerr << "Base64 decode failed: empty result" << std::endl;
            return cv::Mat();
        }
        
        // 使用 imdecode 将字节流转换为图像
        cv::Mat img = cv::imdecode(cv::Mat(decoded_data), cv::IMREAD_COLOR);
        
        if (img.empty()) {
            std::cerr << "Failed to decode image from bytes" << std::endl;
        }
        
        return img;
    } catch (const std::exception& e) {
        std::cerr << "Exception in base64ToMat: " << e.what() << std::endl;
        return cv::Mat();
    }
}

// 初始化识别器（可选，也可以在 ProcessBatchImages 中自动初始化）
extern "C" int FR_Initialize(const char* model_path) {
    try {
        if (g_initialized && g_modelPath == model_path) {
            std::cout << "Model already initialized" << std::endl;
            return 0;
        }
        
        std::cout << "Initializing face recognizer with model: " << model_path << std::endl;
        
        g_recognizer = std::make_unique<FaceRecognizer>();
        
        if (!g_recognizer->loadModel(model_path)) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            g_recognizer.reset();
            g_initialized = false;
            return -1;
        }
        
        g_modelPath = model_path;
        g_initialized = true;
        std::cout << "Face recognizer initialized successfully" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception in FR_Initialize: " << e.what() << std::endl;
        return -2;
    }
}

// 核心处理接口：批量处理 Base64 图片
extern "C" int FR_ProcessBatchImages(
    const BatchImageInput* input,
    BatchImageOutput* output
) {
    if (!input || !output) {
        std::cerr << "Invalid input or output pointer" << std::endl;
        return -1;
    }
    
    if (!input->images || input->count <= 0) {
        std::cerr << "Invalid input images or count" << std::endl;
        return -2;
    }
    
    // 确保识别器已初始化
    if (!g_initialized || !g_recognizer) {
        std::cerr << "Recognizer not initialized. Call FR_Initialize first." << std::endl;
        return -3;
    }
    
    std::cout << "Processing batch of " << input->count << " images" << std::endl;
    
    try {
        // 分配结果数组
        output->count = input->count;
        output->results = new ImageResult[input->count];
        
        // 解码所有图片
        std::vector<cv::Mat> images;
        std::vector<int> validIndices;
        
        for (int i = 0; i < input->count; i++) {
            ImageResult& result = output->results[i];
            const ImageBase64& img_input = input->images[i];
            
            // 初始化结果
            result.features = nullptr;
            result.feature_dim = 0;
            result.status = -1;
            
            // 验证输入
            if (!img_input.base64_str || img_input.str_len <= 0) {
                std::cerr << "  Invalid base64 string for image " << i << std::endl;
                result.status = -10;
                images.push_back(cv::Mat()); // 占位
                continue;
            }
            
            // Base64 解码并转换为 Mat
            cv::Mat image = base64ToMat(img_input.base64_str, img_input.str_len);
            
            if (image.empty()) {
                std::cerr << "  Failed to decode image " << i << std::endl;
                result.status = -11;
                images.push_back(cv::Mat()); // 占位
                continue;
            }
            
            std::cout << "  Image " << i << " decoded: " << image.cols << "x" << image.rows << std::endl;
            images.push_back(image);
            validIndices.push_back(i);
        }
        
        // 批量提取特征
        std::cout << "Running batch feature extraction..." << std::endl;
        auto allFeatures = g_recognizer->extractFeaturesBatchSimple(images);
        
        // 将结果复制到输出
        for (size_t i = 0; i < allFeatures.size() && i < static_cast<size_t>(input->count); i++) {
            ImageResult& result = output->results[i];
            
            if (!allFeatures[i].empty()) {
                // 分配内存并复制特征
                result.feature_dim = allFeatures[i].size();
                result.features = new float[result.feature_dim];
                std::memcpy(result.features, allFeatures[i].data(), result.feature_dim * sizeof(float));
                result.status = 0; // 成功
                
                std::cout << "  Image " << i << " processed successfully (dim: " << result.feature_dim << ")" << std::endl;
            } else {
                result.status = -12; // 特征提取失败
                std::cerr << "  Feature extraction failed for image " << i << std::endl;
            }
        }
        
        std::cout << "Batch processing completed" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in FR_ProcessBatchImages: " << e.what() << std::endl;
        
        // 清理已分配的内存
        if (output->results) {
            for (int i = 0; i < output->count; i++) {
                if (output->results[i].features) {
                    delete[] output->results[i].features;
                }
            }
            delete[] output->results;
            output->results = nullptr;
        }
        
        return -100;
    }
}

// 释放批量结果
extern "C" void FR_FreeBatchResults(BatchImageOutput* output) {
    if (!output) {
        return;
    }
    
    std::cout << "Freeing batch results for " << output->count << " images" << std::endl;
    
    if (output->results) {
        for (int i = 0; i < output->count; i++) {
            if (output->results[i].features) {
                delete[] output->results[i].features;
                output->results[i].features = nullptr;
            }
        }
        delete[] output->results;
        output->results = nullptr;
    }
    
    output->count = 0;
    
    std::cout << "Batch results freed" << std::endl;
}

// 释放识别器资源
extern "C" void FR_Cleanup() {
    std::cout << "Cleaning up face recognizer" << std::endl;
    g_recognizer.reset();
    g_initialized = false;
    g_modelPath.clear();
    std::cout << "Cleanup completed" << std::endl;
}

// 计算两个特征向量的相似度
extern "C" float FR_CompareFaces(
    const float* features1, int dim1,
    const float* features2, int dim2
) {
    if (!features1 || !features2 || dim1 != dim2 || dim1 <= 0) {
        std::cerr << "Invalid feature vectors for comparison" << std::endl;
        return -1.0f;
    }
    
    // 使用识别器的比对函数
    if (g_initialized && g_recognizer) {
        std::vector<float> feat1(features1, features1 + dim1);
        std::vector<float> feat2(features2, features2 + dim2);
        return g_recognizer->compareFaces(feat1, feat2);
    }
    
    // 如果识别器未初始化，直接计算余弦相似度
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (int i = 0; i < dim1; i++) {
        dot_product += features1[i] * features2[i];
        norm1 += features1[i] * features1[i];
        norm2 += features2[i] * features2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < 1e-6 || norm2 < 1e-6) {
        return 0.0f;
    }
    
    // 余弦相似度映射到 [0, 1]
    float cosine = dot_product / (norm1 * norm2);
    return (cosine + 1.0f) / 2.0f;
}

