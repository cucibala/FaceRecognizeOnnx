#include "JniInterface.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
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

// 全局统计数据
struct GlobalStats {
    std::atomic<int> totalProcessed{0};
    std::atomic<int> totalSuccess{0};
    std::atomic<long long> totalLatency{0};
    std::mutex mutex;
};

// 工作线程函数
void workerThread(int threadId, const std::string& base64, int batchSize, int numBatches, 
                  GlobalStats& stats, std::atomic<bool>& stopFlag) {
    for (int i = 0; i < numBatches && !stopFlag; i++) {
        // 准备输入
        std::vector<std::string> base64Strings(batchSize, base64);
        std::vector<ImageBase64> images;
        for (int j = 0; j < batchSize; j++) {
            ImageBase64 img;
            img.base64_str = base64Strings[j].c_str();
            img.str_len = base64Strings[j].length();
            images.push_back(img);
        }
        
        BatchImageInput input;
        input.images = images.data();
        input.count = images.size();
        
        // 调用处理接口（会阻塞等待）
        BatchImageOutput output;
        output.results = nullptr;
        output.count = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        int ret = FR_ProcessBatchImages(&input, &output);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (ret == 0) {
            int successCount = 0;
            for (int j = 0; j < output.count; j++) {
                if (output.results[j].status == 0) successCount++;
            }
            
            stats.totalProcessed += output.count;
            stats.totalSuccess += successCount;
            stats.totalLatency += duration.count();
            
            FR_FreeBatchResults(&output);
        }
    }
}

void runBenchmark(const char* modelPath, const std::string& imagePath, bool useGPU, int deviceId) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Multi-threaded Batch Processing Test" << std::endl;
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
    
    // 转换图片为 Base64
    std::string base64 = imageToBase64(imagePath);
    if (base64.empty()) {
        std::cerr << "Failed to encode image" << std::endl;
        return;
    }
    
    // 测试配置
    struct TestConfig {
        int numThreads;
        int batchSize;
        int totalImages;  // 总共要处理的图片数
    };
    
    std::vector<TestConfig> configs = {
        {1, 16, 1024},   // 1线程，每次16张，总共1024张（测试超时机制）
        {2, 32, 2048},   // 2线程，每次32张，总共2048张
        {4, 32, 4096},   // 4线程，每次32张，总共4096张
        {8, 32, 8192},   // 8线程，每次32张，总共8192张
        {16, 16, 8192},  // 16线程，每次16张，总共8192张
    };
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Running concurrent tests..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    struct BenchmarkResult {
        int numThreads;
        int batchSize;
        int totalImages;
        int successCount;
        long long totalTime;
        double throughput;
        double avgLatency;
    };
    
    std::vector<BenchmarkResult> results;
    
    for (const auto& config : configs) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Test: " << config.numThreads << " threads, batch=" 
                  << config.batchSize << ", total=" << config.totalImages << " images" << std::endl;
        std::cout << "========================================" << std::endl;
        
        GlobalStats stats;
        std::atomic<bool> stopFlag{false};
        
        int numBatchesPerThread = config.totalImages / (config.numThreads * config.batchSize);
        
        std::cout << "Starting " << config.numThreads << " worker threads..." << std::endl;
        std::cout << "Each thread will process " << numBatchesPerThread << " batches of " 
                  << config.batchSize << " images" << std::endl;
        
        auto testStart = std::chrono::high_resolution_clock::now();
        
        // 启动工作线程
        std::vector<std::thread> threads;
        for (int i = 0; i < config.numThreads; i++) {
            threads.emplace_back(workerThread, i, std::ref(base64), config.batchSize, 
                               numBatchesPerThread, std::ref(stats), std::ref(stopFlag));
        }
        
        // 等待所有线程完成
        for (auto& t : threads) {
            t.join();
        }
        
        auto testEnd = std::chrono::high_resolution_clock::now();
        auto testDuration = std::chrono::duration_cast<std::chrono::milliseconds>(testEnd - testStart);
        
        // 强制处理缓冲区中的剩余图片
        std::cout << "Flushing remaining images..." << std::endl;
        FR_FlushBatch();
        
        // 统计结果
        BenchmarkResult result;
        result.numThreads = config.numThreads;
        result.batchSize = config.batchSize;
        result.totalImages = stats.totalProcessed.load();
        result.successCount = stats.totalSuccess.load();
        result.totalTime = testDuration.count();
        result.throughput = result.totalTime > 0 ? result.successCount * 1000.0 / result.totalTime : 0;
        result.avgLatency = result.successCount > 0 ? stats.totalLatency.load() * 1.0 / (stats.totalProcessed.load() / config.batchSize) : 0;
        results.push_back(result);
        
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Total processed: " << result.totalImages << " images" << std::endl;
        std::cout << "  Success count: " << result.successCount << " images" << std::endl;
        std::cout << "  Total time: " << result.totalTime << " ms" << std::endl;
        std::cout << "  Throughput: " << result.throughput << " img/s" << std::endl;
        std::cout << "  Avg latency per batch: " << result.avgLatency << " ms" << std::endl;
    }
    
    // 清理
    std::cout << "\nCleaning up..." << std::endl;
    FR_Cleanup();
    
    // 输出总结
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    printf("%-10s %-12s %-15s %-15s %-18s %-18s\n", 
           "Threads", "Batch Size", "Total Images", "Success", "Time(ms)", "Throughput(img/s)");
    printf("%-10s %-12s %-15s %-15s %-18s %-18s\n", 
           "-------", "----------", "------------", "-------", "--------", "-----------------");
    
    for (const auto& result : results) {
        printf("%-10d %-12d %-15d %-15d %-18lld %-18.2f\n",
               result.numThreads,
               result.batchSize,
               result.totalImages,
               result.successCount,
               result.totalTime,
               result.throughput);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Throughput Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (results.size() >= 2) {
        double baseline = results[0].throughput;
        std::cout << "Baseline (" << results[0].numThreads << " threads): " 
                  << baseline << " img/s" << std::endl;
        std::cout << std::endl;
        
        for (size_t i = 1; i < results.size(); i++) {
            double improvement = (results[i].throughput - baseline) / baseline * 100;
            
            std::cout << results[i].numThreads << " threads vs " << results[0].numThreads << " threads:" << std::endl;
            std::cout << "  Throughput: " << results[i].throughput << " img/s" << std::endl;
            std::cout << "  Change: " << (improvement >= 0 ? "+" : "") << improvement << "%" << std::endl;
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
        std::cout << "  This tool tests multi-threaded batch processing performance." << std::endl;
        std::cout << "  It uses concurrent threads to continuously submit images to test the" << std::endl;
        std::cout << "  smart batching mechanism (32 images batch + 20ms timeout)." << std::endl;
        std::cout << "\nTest Configurations:" << std::endl;
        std::cout << "  - 1 thread,   batch=16, total=1024 images  (tests timeout mechanism)" << std::endl;
        std::cout << "  - 2 threads,  batch=32, total=2048 images" << std::endl;
        std::cout << "  - 4 threads,  batch=32, total=4096 images" << std::endl;
        std::cout << "  - 8 threads,  batch=32, total=8192 images" << std::endl;
        std::cout << "  - 16 threads, batch=16, total=8192 images" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  --gpu         Enable GPU acceleration (requires GPU build)" << std::endl;
        std::cout << "  --device <id> Specify GPU device ID (default: 0)" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " models/w600k_mbf.onnx test.jpg --gpu" << std::endl;
        std::cout << "  " << argv[0] << " models/w600k_mbf.onnx test.jpg --gpu --device 1" << std::endl;
        std::cout << "\nOutput:" << std::endl;
        std::cout << "  - Total throughput (images/second)" << std::endl;
        std::cout << "  - Average latency per batch" << std::endl;
        std::cout << "  - Comparison across different thread counts" << std::endl;
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

