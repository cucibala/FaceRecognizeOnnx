#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "face_detector.h"
#include "face_recognizer.h"

void drawFaceInfo(cv::Mat& image, const FaceBox& face, const std::string& label = "", float similarity = -1.0f) {
    // 绘制人脸框
    cv::rectangle(image, face.box, cv::Scalar(0, 255, 0), 2);
    
    // 绘制关键点
    for (int i = 0; i < 5; i++) {
        cv::circle(image, face.landmarks[i], 2, cv::Scalar(0, 0, 255), -1);
    }
    
    // 绘制置信度和相似度信息
    std::string text = "Score: " + std::to_string(face.score).substr(0, 5);
    if (similarity >= 0) {
        text += " | Sim: " + std::to_string(similarity).substr(0, 5);
    }
    if (!label.empty()) {
        text = label + " | " + text;
    }
    
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    
    // 绘制文字背景
    cv::rectangle(image, 
                  cv::Point(face.box.x, face.box.y - textSize.height - 10),
                  cv::Point(face.box.x + textSize.width, face.box.y),
                  cv::Scalar(0, 255, 0), -1);
    
    // 绘制文字
    cv::putText(image, text, 
                cv::Point(face.box.x, face.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

void testDetection(FaceDetector& detector, const std::string& imagePath) {
    std::cout << "\n=== 测试人脸检测 ===" << std::endl;
    
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << imagePath << std::endl;
        std::cerr << "请检查文件路径是否正确，文件是否存在" << std::endl;
        return;
    }
    
    std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
    
    auto faces = detector.detect(image);
    std::cout << "检测到 " << faces.size() << " 个人脸" << std::endl;
    
    for (size_t i = 0; i < faces.size(); i++) {
        std::cout << "人脸 " << i + 1 << ": "
                  << "位置(" << faces[i].box.x << ", " << faces[i].box.y << ", "
                  << faces[i].box.width << ", " << faces[i].box.height << ") "
                  << "置信度: " << faces[i].score << std::endl;
        
        drawFaceInfo(image, faces[i]);
    }
    
    cv::imshow("人脸检测结果", image);
    cv::waitKey(0);
}

void testRecognition(FaceDetector& detector, FaceRecognizer& recognizer, 
                     const std::string& image1Path, const std::string& image2Path) {
    std::cout << "\n=== 测试人脸识别与比对 ===" << std::endl;
    
    cv::Mat image1 = cv::imread(image1Path);
    cv::Mat image2 = cv::imread(image2Path);
    
    if (image1.empty()) {
        std::cerr << "无法读取图像1: " << image1Path << std::endl;
        return;
    }
    
    if (image2.empty()) {
        std::cerr << "无法读取图像2: " << image2Path << std::endl;
        return;
    }
    
    std::cout << "图像1尺寸: " << image1.cols << "x" << image1.rows << std::endl;
    std::cout << "图像2尺寸: " << image2.cols << "x" << image2.rows << std::endl;
    
    // 检测人脸
    auto faces1 = detector.detect(image1);
    auto faces2 = detector.detect(image2);
    
    if (faces1.empty() || faces2.empty()) {
        std::cerr << "未检测到人脸" << std::endl;
        return;
    }
    
    std::cout << "图像1检测到 " << faces1.size() << " 个人脸" << std::endl;
    std::cout << "图像2检测到 " << faces2.size() << " 个人脸" << std::endl;
    
    // 提取特征
    std::cout << "提取图像1的人脸特征..." << std::endl;
    auto feature1 = recognizer.extractFeature(image1, faces1[0]);
    
    std::cout << "提取图像2的人脸特征..." << std::endl;
    auto feature2 = recognizer.extractFeature(image2, faces2[0]);
    
    if (feature1.empty() || feature2.empty()) {
        std::cerr << "特征提取失败" << std::endl;
        return;
    }
    
    std::cout << "特征维度: " << feature1.size() << std::endl;
    
    // 比对人脸
    float similarity = recognizer.compareFaces(feature1, feature2);
    std::cout << "相似度: " << similarity << std::endl;
    
    // 判断是否为同一人
    float threshold = 0.6f; // 可调整阈值
    if (similarity > threshold) {
        std::cout << "结果: 同一人 (相似度: " << similarity << " > " << threshold << ")" << std::endl;
    } else {
        std::cout << "结果: 不同人 (相似度: " << similarity << " <= " << threshold << ")" << std::endl;
    }
    
    // 显示结果
    drawFaceInfo(image1, faces1[0], "Image 1");
    drawFaceInfo(image2, faces2[0], "Image 2", similarity);
    
    cv::Mat combined;
    cv::hconcat(image1, image2, combined);
    
    cv::imshow("人脸比对结果", combined);
    cv::waitKey(0);
}

void testRecognitionSimple(FaceRecognizer& recognizer, 
                           const std::string& image1Path, const std::string& image2Path) {
    std::cout << "\n=== 测试人脸识别与比对（简化模式 - 无检测） ===" << std::endl;
    
    cv::Mat image1 = cv::imread(image1Path);
    cv::Mat image2 = cv::imread(image2Path);
    
    if (image1.empty()) {
        std::cerr << "无法读取图像1: " << image1Path << std::endl;
        return;
    }
    
    if (image2.empty()) {
        std::cerr << "无法读取图像2: " << image2Path << std::endl;
        return;
    }
    
    std::cout << "\n处理图像1..." << std::endl;
    std::cout << "原始尺寸: " << image1.cols << "x" << image1.rows << std::endl;
    auto feature1 = recognizer.extractFeatureSimple(image1);
    
    std::cout << "\n处理图像2..." << std::endl;
    std::cout << "原始尺寸: " << image2.cols << "x" << image2.rows << std::endl;
    auto feature2 = recognizer.extractFeatureSimple(image2);
    
    if (feature1.empty() || feature2.empty()) {
        std::cerr << "\n特征提取失败" << std::endl;
        return;
    }
    
    std::cout << "\n特征维度: " << feature1.size() << std::endl;
    
    // 比对人脸
    float similarity = recognizer.compareFaces(feature1, feature2);
    std::cout << "\n相似度: " << similarity << std::endl;
    
    // 判断是否为同一人
    float threshold = 0.6f;
    if (similarity > threshold) {
        std::cout << "结果: 同一人 (相似度: " << similarity << " > " << threshold << ")" << std::endl;
    } else {
        std::cout << "结果: 不同人 (相似度: " << similarity << " <= " << threshold << ")" << std::endl;
    }
}

void testBatchProcessing(FaceRecognizer& recognizer, const std::vector<std::string>& imagePaths) {
    std::cout << "\n=== 测试批量处理 ===" << std::endl;
    
    // 加载所有图片
    std::vector<cv::Mat> images;
    std::cout << "Loading " << imagePaths.size() << " images..." << std::endl;
    
    for (size_t i = 0; i < imagePaths.size(); i++) {
        cv::Mat img = cv::imread(imagePaths[i]);
        if (img.empty()) {
            std::cerr << "  Failed to load: " << imagePaths[i] << std::endl;
            images.push_back(cv::Mat()); // 添加空图占位
        } else {
            std::cout << "  Loaded " << i << ": " << imagePaths[i] 
                      << " (" << img.cols << "x" << img.rows << ")" << std::endl;
            images.push_back(img);
        }
    }
    
    if (images.empty()) {
        std::cerr << "No valid images to process" << std::endl;
        return;
    }
    
    // 批量提取特征
    std::cout << "\nExtracting features in batch mode..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto features = recognizer.extractFeaturesBatchSimple(images);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\nBatch processing completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Average time per image: " 
              << (features.empty() ? 0 : duration.count() / features.size()) << " ms" << std::endl;
    
    // 显示结果
    std::cout << "\nResults:" << std::endl;
    int successCount = 0;
    for (size_t i = 0; i < features.size(); i++) {
        std::cout << "  Image " << i << ": ";
        if (!features[i].empty()) {
            std::cout << "SUCCESS (dim: " << features[i].size() << ")" << std::endl;
            successCount++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    }
    
    std::cout << "\nSuccess rate: " << successCount << "/" << features.size() << std::endl;
    
    // 如果有至少2个成功的结果，进行比对
    if (successCount >= 2) {
        std::cout << "\nComputing pairwise similarities:" << std::endl;
        
        // 找出所有成功的索引
        std::vector<int> validIndices;
        for (size_t i = 0; i < features.size(); i++) {
            if (!features[i].empty()) {
                validIndices.push_back(i);
            }
        }
        
        // 计算前几对的相似度
        int maxPairs = std::min(5, static_cast<int>(validIndices.size() * (validIndices.size() - 1) / 2));
        int pairCount = 0;
        
        for (size_t i = 0; i < validIndices.size() && pairCount < maxPairs; i++) {
            for (size_t j = i + 1; j < validIndices.size() && pairCount < maxPairs; j++) {
                int idx1 = validIndices[i];
                int idx2 = validIndices[j];
                
                float similarity = recognizer.compareFaces(features[idx1], features[idx2]);
                std::cout << "  Image " << idx1 << " vs Image " << idx2 
                          << ": " << similarity;
                
                if (similarity > 0.6f) {
                    std::cout << " (Same person)";
                } else {
                    std::cout << " (Different)";
                }
                std::cout << std::endl;
                
                pairCount++;
            }
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "InsightFace C++ Demo - buffalo_sc 模型" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 模型路径
    std::string detectorModelPath = "models/det_500m.onnx";
    std::string recognizerModelPath = "models/w600k_mbf.onnx";
    
    // 加载检测模型
    FaceDetector detector;
    if (!detector.loadModel(detectorModelPath)) {
        std::cerr << "无法加载人脸检测模型: " << detectorModelPath << std::endl;
        return -1;
    }
    
    // 加载识别模型
    FaceRecognizer recognizer;
    if (!recognizer.loadModel(recognizerModelPath)) {
        std::cerr << "无法加载人脸识别模型: " << recognizerModelPath << std::endl;
        return -1;
    }
    
    std::cout << "\n所有模型加载成功!" << std::endl;
    
    // 使用示例
    if (argc < 2) {
        std::cout << "\n使用方法:" << std::endl;
        std::cout << "1. 人脸检测: " << argv[0] << " detect <image_path>" << std::endl;
        std::cout << "2. 人脸比对: " << argv[0] << " compare <image1_path> <image2_path>" << std::endl;
        std::cout << "3. 简化比对: " << argv[0] << " simple <image1_path> <image2_path>" << std::endl;
        std::cout << "4. 批量处理: " << argv[0] << " batch <image1> <image2> [image3] ..." << std::endl;
        std::cout << "\n示例:" << std::endl;
        std::cout << "  " << argv[0] << " detect test.jpg" << std::endl;
        std::cout << "  " << argv[0] << " compare person1.jpg person2.jpg" << std::endl;
        std::cout << "  " << argv[0] << " simple person1.jpg person2.jpg  # 直接resize，不检测人脸" << std::endl;
        std::cout << "  " << argv[0] << " batch img1.jpg img2.jpg img3.jpg  # 批量处理" << std::endl;
        return 0;
    }
    
    std::string mode = argv[1];
    
    if (mode == "detect" && argc >= 3) {
        testDetection(detector, argv[2]);
    } else if (mode == "compare" && argc >= 4) {
        testRecognition(detector, recognizer, argv[2], argv[3]);
    } else if (mode == "simple" && argc >= 4) {
        testRecognitionSimple(recognizer, argv[2], argv[3]);
    } else if (mode == "batch" && argc >= 3) {
        std::vector<std::string> imagePaths;
        for (int i = 2; i < argc; i++) {
            imagePaths.push_back(argv[i]);
        }
        testBatchProcessing(recognizer, imagePaths);
    } else {
        std::cerr << "无效的命令或参数" << std::endl;
        return -1;
    }
    
    return 0;
}

