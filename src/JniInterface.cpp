#include "JniInterface.h"
#include "face_recognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <chrono>

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

// SIMD 优化的 Base64 解码查找表
static const uint8_t base64_decode_table[256] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,  255, 255, 255, 63,
    52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255, 255, 254, 255, 255,
    255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  255, 255, 255, 255, 255,
    255, 26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
    41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

// 优化的 Base64 解码（使用查找表 + 批量处理）
std::vector<unsigned char> base64_decode(const std::string& encoded_string) {
    const size_t in_len = encoded_string.size();
    if (in_len == 0) return std::vector<unsigned char>();
    
    // 预分配输出缓冲区（避免多次 realloc）
    std::vector<unsigned char> ret;
    ret.reserve((in_len * 3) / 4 + 3);  // 预留足够空间
    
    size_t i = 0;
    
    // 快速路径：批量处理 4 字节块
    while (i + 4 <= in_len) {
        // 检查 padding 或结束
        if (encoded_string[i] == '=' || encoded_string[i] == '\0') break;
        
        // 查表获取 4 个 6-bit 值
        uint8_t b0 = base64_decode_table[static_cast<uint8_t>(encoded_string[i])];
        uint8_t b1 = base64_decode_table[static_cast<uint8_t>(encoded_string[i+1])];
        uint8_t b2 = base64_decode_table[static_cast<uint8_t>(encoded_string[i+2])];
        uint8_t b3 = base64_decode_table[static_cast<uint8_t>(encoded_string[i+3])];
        
        // 检查非法字符（255 表示非法）
        if (b0 == 255 || b1 == 255) break;
        
        // 第一个输出字节（总是存在）
        ret.push_back((b0 << 2) | (b1 >> 4));
        
        // 第二个输出字节（如果不是 padding）
        if (b2 != 254) {  // 254 表示 '='
            ret.push_back((b1 << 4) | (b2 >> 2));
            
            // 第三个输出字节（如果不是 padding）
            if (b3 != 254) {
                ret.push_back((b2 << 6) | b3);
            }
        }
        
        i += 4;
    }
    
    return ret;
}

// 将 Base64 字符串转换为 OpenCV Mat
cv::Mat base64ToMat(const char* base64_str, int str_len) {
    try {
        auto t_base64_start = std::chrono::high_resolution_clock::now();

        std::string encoded_string(base64_str, str_len);
        std::vector<unsigned char> decoded_data = base64_decode(encoded_string);

        auto t_base64_end = std::chrono::high_resolution_clock::now();
        double base64_decode_ms = std::chrono::duration<double, std::milli>(t_base64_end - t_base64_start).count();
        std::cout << "[Time] Base64 decode: " << base64_decode_ms << " ms (decoded " << decoded_data.size() << " bytes)" << std::endl;
        
        if (decoded_data.empty()) {
            std::cerr << "Base64 decode failed: empty result (input length: " << str_len << ")" << std::endl;
            return cv::Mat();
        }

        auto t_imdecode_start = std::chrono::high_resolution_clock::now();
        // 使用 imdecode 将字节流转换为图像
        cv::Mat img = cv::imdecode(cv::Mat(decoded_data), cv::IMREAD_COLOR);
        auto t_imdecode_end = std::chrono::high_resolution_clock::now();
        double imdecode_ms = std::chrono::duration<double, std::milli>(t_imdecode_end - t_imdecode_start).count();
        std::cout << "[Time] imdecode: " << imdecode_ms << " ms" << std::endl;

        if (img.empty()) {
            std::cerr << "Failed to decode image from bytes (decoded " << decoded_data.size() 
                      << " bytes, first 16 bytes: ";
            for (size_t i = 0; i < std::min(size_t(16), decoded_data.size()); i++) {
                std::cerr << std::hex << (int)decoded_data[i] << " ";
            }
            std::cerr << std::dec << ")" << std::endl;
        }
        
        return img;
    } catch (const std::exception& e) {
        std::cerr << "Exception in base64ToMat: " << e.what() << std::endl;
        return cv::Mat();
    }
}

// 初始化识别器（可选，也可以在 ProcessBatchImages 中自动初始化）
extern "C" int FR_Initialize(const char* model_path) {
    return FR_InitializeWithGPU(model_path, false, 0);
}

// 初始化识别器（带 GPU 选项）
extern "C" int FR_InitializeWithGPU(const char* model_path, bool use_gpu, int device_id) {
    try {
        if (g_initialized && g_modelPath == model_path) {
            return 0;
        }
        
        std::cout << "Initializing: " << model_path << " [" << (use_gpu ? "GPU" : "CPU") << "]" << std::endl;
        
        g_recognizer = std::make_unique<FaceRecognizer>(use_gpu, device_id);
        
        if (!g_recognizer->loadModel(model_path)) {
            std::cerr << "Failed to load model" << std::endl;
            g_recognizer.reset();
            g_initialized = false;
            return -1;
        }
        
        g_modelPath = model_path;
        g_initialized = true;
        
        // TensorRT 引擎预热（提前构建常用 batch size 的引擎）
        if (use_gpu) {
            g_recognizer->warmupTensorRT({1, 16, 32});
        }
        
        std::cout << "Ready" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Init failed: " << e.what() << std::endl;
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
    
    try {
        auto decodeStart = std::chrono::high_resolution_clock::now();
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
                result.status = -10;
                images.push_back(cv::Mat()); // 占位
                continue;
            }
            
            // Base64 解码并转换为 Mat
            cv::Mat image = base64ToMat(img_input.base64_str, img_input.str_len);
            
            if (image.empty()) {
                result.status = -11;
                images.push_back(cv::Mat()); // 占位
                continue;
            }
            
            images.push_back(image);
            validIndices.push_back(i);
        }
        
        auto decodeEnd = std::chrono::high_resolution_clock::now();
        auto decodeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(decodeEnd - decodeStart);
        std::cout << "Decode time: " << decodeDuration.count() << " ms" << std::endl;

        auto extractStart = std::chrono::high_resolution_clock::now();
        // 批量提取特征
        auto allFeatures = g_recognizer->extractFeaturesBatchSimple(images);
        auto extractEnd = std::chrono::high_resolution_clock::now();
        auto extractDuration = std::chrono::duration_cast<std::chrono::milliseconds>(extractEnd - extractStart);
        std::cout << "Extract time: " << extractDuration.count() << " ms" << std::endl;

        // 将结果复制到输出
        for (size_t i = 0; i < allFeatures.size() && i < static_cast<size_t>(input->count); i++) {
            ImageResult& result = output->results[i];
            
            if (!allFeatures[i].empty()) {
                // 分配内存并复制特征
                result.feature_dim = allFeatures[i].size();
                result.features = new float[result.feature_dim];
                std::memcpy(result.features, allFeatures[i].data(), result.feature_dim * sizeof(float));
                result.status = 0; // 成功
            } else {
                result.status = -12; // 特征提取失败
            }
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Batch processing error: " << e.what() << std::endl;
        
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
}

// 释放识别器资源
extern "C" void FR_Cleanup() {
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

