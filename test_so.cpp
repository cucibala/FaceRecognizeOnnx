#include "src/JniInterface.h"
#include <iostream>
#include <fstream>
#include <vector>

// 简单测试加载 .so 库
int main() {
    std::cout << "Testing libimage_processor.so" << std::endl;
    std::cout << "================================" << std::endl;
    
    // 1. 初始化
    const char* modelPath = "models/w600k_mbf.onnx";
    std::cout << "\n1. Initializing with model: " << modelPath << std::endl;
    
    int ret = FR_Initialize(modelPath);
    if (ret != 0) {
        std::cerr << "Initialization failed: " << ret << std::endl;
        return 1;
    }
    
    std::cout << "✓ Initialization successful!" << std::endl;
    
    // 2. 清理
    std::cout << "\n2. Cleaning up..." << std::endl;
    FR_Cleanup();
    std::cout << "✓ Cleanup successful!" << std::endl;
    
    std::cout << "\n============================" << std::endl;
    std::cout << "Library test passed!" << std::endl;
    
    return 0;
}

