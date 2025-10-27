#include "JniInterface.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

// 将图片文件转换为 Base64
std::string imageToBase64(const std::string& imagePath) {
    // 读取图片文件
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return "";
    }
    
    // 将图片编码为 JPEG 字节流
    std::vector<uchar> buf;
    cv::imencode(".jpg", image, buf);
    
    // Base64 编码
    const std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    int in_len = buf.size();
    const unsigned char* bytes_to_encode = buf.data();
    
    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            
            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    
    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';
        
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        
        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];
        
        while((i++ < 3))
            ret += '=';
    }
    
    return ret;
}

void testJniInterface(const char* modelPath, const std::vector<std::string>& imagePaths) {
    std::cout << "=== JNI Interface Test ===" << std::endl;
    
    // 1. 初始化
    std::cout << "\n1. Initializing..." << std::endl;
    int ret = FR_Initialize(modelPath);
    if (ret != 0) {
        std::cerr << "Initialization failed with code: " << ret << std::endl;
        return;
    }
    
    // 2. 准备输入数据
    std::cout << "\n2. Preparing input data..." << std::endl;
    std::vector<std::string> base64Strings;
    std::vector<ImageBase64> images;
    
    for (size_t i = 0; i < imagePaths.size(); i++) {
        std::cout << "  Converting image " << (i + 1) << ": " << imagePaths[i] << std::endl;
        std::string base64 = imageToBase64(imagePaths[i]);
        
        if (base64.empty()) {
            std::cerr << "  Failed to convert image to base64" << std::endl;
            continue;
        }
        
        base64Strings.push_back(base64);
        
        ImageBase64 img;
        img.base64_str = base64Strings.back().c_str();
        img.str_len = base64Strings.back().length();
        img.user_data = nullptr;
        
        images.push_back(img);
        
        std::cout << "  Base64 length: " << img.str_len << std::endl;
    }
    
    if (images.empty()) {
        std::cerr << "No valid images to process" << std::endl;
        FR_Cleanup();
        return;
    }
    
    BatchImageInput input;
    input.images = images.data();
    input.count = images.size();
    
    // 3. 批量处理
    std::cout << "\n3. Processing batch..." << std::endl;
    BatchImageOutput output;
    output.results = nullptr;
    output.count = 0;
    
    ret = FR_ProcessBatchImages(&input, &output);
    if (ret != 0) {
        std::cerr << "Batch processing failed with code: " << ret << std::endl;
        FR_Cleanup();
        return;
    }
    
    // 4. 查看结果
    std::cout << "\n4. Processing results:" << std::endl;
    int successCount = 0;
    
    for (int i = 0; i < output.count; i++) {
        const ImageResult& result = output.results[i];
        
        std::cout << "  Image " << (i + 1) << ": ";
        if (result.status == 0) {
            std::cout << "SUCCESS - Feature dim: " << result.feature_dim << std::endl;
            
            // 打印前 10 个特征值
            std::cout << "    First 10 features: ";
            for (int j = 0; j < std::min(10, result.feature_dim); j++) {
                std::cout << result.features[j] << " ";
            }
            std::cout << std::endl;
            
            successCount++;
        } else {
            std::cout << "FAILED - Status: " << result.status << std::endl;
        }
    }
    
    std::cout << "\nSuccess rate: " << successCount << "/" << output.count << std::endl;
    
    // 5. 如果有多个成功的结果，计算相似度
    if (successCount >= 2) {
        std::cout << "\n5. Computing similarities:" << std::endl;
        
        // 找出前两个成功的结果
        int idx1 = -1, idx2 = -1;
        for (int i = 0; i < output.count && idx2 == -1; i++) {
            if (output.results[i].status == 0) {
                if (idx1 == -1) {
                    idx1 = i;
                } else {
                    idx2 = i;
                }
            }
        }
        
        if (idx1 >= 0 && idx2 >= 0) {
            float similarity = FR_CompareFaces(
                output.results[idx1].features, output.results[idx1].feature_dim,
                output.results[idx2].features, output.results[idx2].feature_dim
            );
            
            std::cout << "  Similarity between image " << (idx1 + 1) 
                      << " and image " << (idx2 + 1) << ": " << similarity << std::endl;
            
            float threshold = 0.6f;
            if (similarity > threshold) {
                std::cout << "  -> Same person (similarity > " << threshold << ")" << std::endl;
            } else {
                std::cout << "  -> Different persons (similarity <= " << threshold << ")" << std::endl;
            }
        }
    }
    
    // 6. 释放结果
    std::cout << "\n6. Freeing results..." << std::endl;
    FR_FreeBatchResults(&output);
    
    // 7. 清理
    std::cout << "\n7. Cleanup..." << std::endl;
    FR_Cleanup();
    
    std::cout << "\n=== Test completed ===" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image1> [image2] [image3] ..." << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " models/w600k_mbf.onnx test1.jpg test2.jpg" << std::endl;
        return 1;
    }
    
    const char* modelPath = argv[1];
    std::vector<std::string> imagePaths;
    
    for (int i = 2; i < argc; i++) {
        imagePaths.push_back(argv[i]);
    }
    
    testJniInterface(modelPath, imagePaths);
    
    return 0;
}

