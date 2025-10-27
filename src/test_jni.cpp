#include "JniInterface.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
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

void runBenchmark(const char* modelPath, const std::string& imagePath, bool useGPU, int deviceId) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Batch Processing Performance Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Image: " << imagePath << std::endl;
    std::cout << "Device: " << (useGPU ? ("GPU " + std::to_string(deviceId)) : "CPU") << std::endl;
    
    // 初始化
    std::cout << "\nInitializing model..." << std::endl;
    int ret = FR_InitializeWithGPU(modelPath, useGPU, deviceId);
    if (ret != 0) {
        std::cerr << "Initialization failed with code: " << ret << std::endl;
        return;
    }
    std::cout << "Model initialized successfully!" << std::endl;
    
    // 测试不同批次大小
    std::vector<int> batchSizes = {1, 16, 16, 16, 16, 16};
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Running performance tests..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    struct BenchmarkResult {
        int batchSize;
        long long totalTime;
        double avgTimePerImage;
        double throughput;
    };
    
    std::vector<BenchmarkResult> results;
    
    for (int batchSize : batchSizes) {
        // 重新运行一次获取准确结果（排除首次缓存影响）
        std::cout << "\n  Running again for accurate measurement..." << std::endl;
        
        std::string base64 = imageToBase64(imagePath);
        std::vector<std::string> base64Strings(batchSize, base64);
        std::vector<ImageBase64> images;
        
        for (int i = 0; i < batchSize; i++) {
            ImageBase64 img;
            img.base64_str = base64Strings[i].c_str();
            img.str_len = base64Strings[i].length();
            images.push_back(img);
        }
        
        BatchImageInput input;
        input.images = images.data();
        input.count = images.size();
        
        BatchImageOutput output;
        output.results = nullptr;
        output.count = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        FR_ProcessBatchImages(&input, &output);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        int successCount = 0;
        for (int i = 0; i < output.count; i++) {
            if (output.results[i].status == 0) successCount++;
        }
        
        BenchmarkResult result;
        result.batchSize = batchSize;
        result.totalTime = duration.count();
        result.avgTimePerImage = successCount > 0 ? duration.count() * 1.0 / successCount : 0;
        result.throughput = duration.count() > 0 ? successCount * 1000.0 / duration.count() : 0;
        results.push_back(result);
        
        FR_FreeBatchResults(&output);
        
        std::cout << "  Confirmed: " << duration.count() << " ms" << std::endl;
    }
    
    // 清理
    std::cout << "\nCleaning up..." << std::endl;
    FR_Cleanup();
    
    // 输出总结
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    printf("%-12s %-15s %-20s %-20s\n", 
           "Batch Size", "Total Time(ms)", "Avg Time/Image(ms)", "Throughput(img/s)");
    printf("%-12s %-15s %-20s %-20s\n", 
           "----------", "--------------", "------------------", "-----------------");
    
    for (const auto& result : results) {
        printf("%-12d %-15lld %-20.2f %-20.2f\n",
               result.batchSize,
               result.totalTime,
               result.avgTimePerImage,
               result.throughput);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Speedup Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (results.size() >= 2) {
        double baseline = results[0].avgTimePerImage;
        std::cout << "Baseline (batch=1): " << baseline << " ms/image" << std::endl;
        std::cout << std::endl;
        
        for (size_t i = 1; i < results.size(); i++) {
            double speedup = baseline / results[i].avgTimePerImage;
            double improvement = (1.0 - results[i].avgTimePerImage / baseline) * 100;
            
            std::cout << "Batch " << results[i].batchSize << " vs Batch 1:" << std::endl;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
            std::cout << "  Improvement: " << improvement << "%" << std::endl;
            std::cout << std::endl;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Test completed!" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path> [--gpu] [--device <id>]" << std::endl;
        std::cout << "\nDescription:" << std::endl;
        std::cout << "  This tool tests batch processing performance with different batch sizes." << std::endl;
        std::cout << "  It will test batch sizes: 1, 16, and 32" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  --gpu         Enable GPU acceleration (requires GPU build)" << std::endl;
        std::cout << "  --device <id> Specify GPU device ID (default: 0)" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " models/w600k_mbf.onnx test.jpg" << std::endl;
        std::cout << "  " << argv[0] << " models/w600k_mbf.onnx test.jpg --gpu" << std::endl;
        std::cout << "  " << argv[0] << " models/w600k_mbf.onnx test.jpg --gpu --device 1" << std::endl;
        std::cout << "\nOutput:" << std::endl;
        std::cout << "  - Processing time for each batch size" << std::endl;
        std::cout << "  - Average time per image" << std::endl;
        std::cout << "  - Throughput (images/second)" << std::endl;
        std::cout << "  - Speedup comparison" << std::endl;
        return 1;
    }
    
    const char* modelPath = argv[1];
    const char* imagePath = argv[2];
    
    bool useGPU = false;
    int deviceId = 0;
    
    // 解析命令行参数
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            useGPU = true;
        } else if (arg == "--device" && i + 1 < argc) {
            deviceId = std::atoi(argv[++i]);
        }
    }
    
    runBenchmark(modelPath, imagePath, useGPU, deviceId);
    
    return 0;
}

