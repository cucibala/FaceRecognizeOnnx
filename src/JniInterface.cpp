#include "JniInterface.h"
#include "face_recognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <chrono>
#include <mutex>
#include <condition_variable>

// 全局变量：单例模式的识别器
static std::unique_ptr<FaceRecognizer> g_recognizer = nullptr;
static std::string g_modelPath = "";
static bool g_initialized = false;

// 批处理缓冲区（1024 张图片）+ 同步机制
struct ImageBuffer {
    std::vector<cv::Mat> images;
    std::vector<ImageResult*> result_ptrs;
    std::vector<int> batch_ids;  // 每个图片所属的批次 ID
    int current_batch_id = 0;
    int processed_batch_id = -1;
    
    std::mutex mutex;
    std::condition_variable cv;
    
    void clear() {
        images.clear();
        result_ptrs.clear();
        batch_ids.clear();
    }
    
    bool is_full() const {
        return images.size() >= 64;
    }
    
    int size() const {
        return images.size();
    }
    
    int add_to_batch(const cv::Mat& img, ImageResult* result_ptr) {
        images.push_back(img);
        result_ptrs.push_back(result_ptr);
        batch_ids.push_back(current_batch_id);
        return current_batch_id;  // 返回该图片的批次 ID
    }
    
    void start_new_batch() {
        current_batch_id++;
    }
};

static ImageBuffer g_image_buffer;

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

// 将 Base64 字符串转换为 OpenCV Mat（优化版本）
cv::Mat base64ToMat(const char* base64_str, int str_len, bool show_timing = false) {
    try {
        auto t_base64_start = std::chrono::high_resolution_clock::now();

        std::string encoded_string(base64_str, str_len);
        std::vector<unsigned char> decoded_data = base64_decode(encoded_string);

        auto t_base64_end = std::chrono::high_resolution_clock::now();
        
        if (decoded_data.empty()) {
            std::cerr << "Base64 decode failed: empty result (input length: " << str_len << ")" << std::endl;
            return cv::Mat();
        }

        auto t_imdecode_start = std::chrono::high_resolution_clock::now();
        
        // 优化：避免不必要的复制，直接使用 decoded_data 的内存
        cv::Mat img = cv::imdecode(decoded_data, cv::IMREAD_COLOR);
        
        auto t_imdecode_end = std::chrono::high_resolution_clock::now();
        
        if (show_timing) {
            double base64_decode_ms = std::chrono::duration<double, std::milli>(t_base64_end - t_base64_start).count();
            double imdecode_ms = std::chrono::duration<double, std::milli>(t_imdecode_end - t_imdecode_start).count();
            std::cout << "[Time] Base64: " << base64_decode_ms << " ms, imdecode: " << imdecode_ms << " ms" << std::endl;
        }

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

// 批量解码 Base64 图片（并行优化）
std::vector<cv::Mat> base64ToMatBatch(const ImageBase64* images, int count) {
    std::vector<cv::Mat> results(count);
    
    // 并行解码所有图片
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < count; i++) {
        if (images[i].base64_str && images[i].str_len > 0) {
            results[i] = base64ToMat(images[i].base64_str, images[i].str_len, false);
        }
    }
    
    return results;
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
// 内部函数：执行批量推理（需要在持有锁的情况下调用）
static void processBatchLocked() {
    if (g_image_buffer.images.empty()) return;
    
    int batch_size = g_image_buffer.size();
    std::cout << "\n=== Processing batch of " << batch_size << " images ===" << std::endl;
    
    auto extractStart = std::chrono::high_resolution_clock::now();
    auto allFeatures = g_recognizer->extractFeaturesBatchSimple(g_image_buffer.images);
    auto extractEnd = std::chrono::high_resolution_clock::now();
    auto extractDuration = std::chrono::duration_cast<std::chrono::milliseconds>(extractEnd - extractStart);
    
    std::cout << "Batch inference completed: " << batch_size << " images in " 
              << extractDuration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (batch_size * 1000.0 / extractDuration.count()) << " img/s" << std::endl;
    
    // 将结果复制到输出
    for (size_t i = 0; i < allFeatures.size() && i < g_image_buffer.result_ptrs.size(); i++) {
        ImageResult* result = g_image_buffer.result_ptrs[i];
        
        if (!allFeatures[i].empty()) {
            result->feature_dim = allFeatures[i].size();
            result->features = new float[result->feature_dim];
            std::memcpy(result->features, allFeatures[i].data(), result->feature_dim * sizeof(float));
            result->status = 0;
        } else {
            result->status = -12;
        }
    }
    
    // 更新已处理的批次 ID
    g_image_buffer.processed_batch_id = g_image_buffer.current_batch_id;
    
    // 清空缓冲区
    g_image_buffer.clear();
    
    // 开始新批次
    g_image_buffer.start_new_batch();
    
    std::cout << "=== Batch processing completed ===" << std::endl << std::endl;
}

// 核心处理接口：批量处理 Base64 图片（累积到 1024 张后推理，同步等待）
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
    
    if (!g_initialized || !g_recognizer) {
        std::cerr << "Recognizer not initialized" << std::endl;
        return -3;
    }
    
    try {
        // 分配结果数组
        output->count = input->count;
        output->results = new ImageResult[input->count];
        
        // 初始化所有结果状态
        for (int i = 0; i < input->count; i++) {
            output->results[i].features = nullptr;
            output->results[i].feature_dim = 0;
            output->results[i].status = -1;
        }
        
        // 并行解码所有图片
        auto decodeStart = std::chrono::high_resolution_clock::now();
        std::vector<cv::Mat> images = base64ToMatBatch(input->images, input->count);
        auto decodeEnd = std::chrono::high_resolution_clock::now();
        auto decodeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(decodeEnd - decodeStart);
        std::cout << "Decode: " << input->count << " images in " << decodeDuration.count() << " ms" << std::endl;
        
        int my_batch_id = -1;
        bool should_process = false;
        
        // 加锁操作缓冲区
        {
            std::unique_lock<std::mutex> lock(g_image_buffer.mutex);
            
            // 将图片添加到缓冲区
            for (int i = 0; i < input->count; i++) {
                if (!input->images[i].base64_str || input->images[i].str_len <= 0) {
                    output->results[i].status = -10;
                } else if (images[i].empty()) {
                    output->results[i].status = -11;
                } else {
                    my_batch_id = g_image_buffer.add_to_batch(images[i], &output->results[i]);
                }
            }
            
            std::cout << "Buffer: " << g_image_buffer.size() << "/1024 images (batch_id=" << my_batch_id << ")" << std::endl;
            
            // 检查是否需要处理
            if (g_image_buffer.is_full()) {
                should_process = true;
            }
        }
        
        // 如果缓冲区满了，执行推理
        if (should_process) {
            std::unique_lock<std::mutex> lock(g_image_buffer.mutex);
            std::cout << "Buffer full! Processing batch..." << std::endl;
            processBatchLocked();
            
            // 通知所有等待的线程
            g_image_buffer.cv.notify_all();
        }
        
        // 等待直到当前批次被处理
        if (my_batch_id >= 0) {
            std::unique_lock<std::mutex> lock(g_image_buffer.mutex);
            g_image_buffer.cv.wait(lock, [my_batch_id]() {
                return g_image_buffer.processed_batch_id >= my_batch_id;
            });
            std::cout << "Batch " << my_batch_id << " processed, returning results" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Batch processing error: " << e.what() << std::endl;
        
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

// 强制处理缓冲区中的剩余图片
extern "C" int FR_FlushBatch() {
    if (!g_initialized || !g_recognizer) {
        std::cerr << "Recognizer not initialized" << std::endl;
        return -3;
    }
    
    std::unique_lock<std::mutex> lock(g_image_buffer.mutex);
    
    if (g_image_buffer.size() == 0) {
        std::cout << "Buffer is empty, nothing to flush" << std::endl;
        return 0;
    }
    
    std::cout << "Flushing buffer: " << g_image_buffer.size() << " images..." << std::endl;
    processBatchLocked();
    
    // 通知所有等待的线程
    g_image_buffer.cv.notify_all();
    
    std::cout << "Buffer flushed" << std::endl;
    
    return 0;
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

