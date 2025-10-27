#include <iostream>
#include <opencv2/opencv.hpp>
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
        return;
    }
    
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
    
    if (image1.empty() || image2.empty()) {
        std::cerr << "无法读取图像" << std::endl;
        return;
    }
    
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

void testWebcam(FaceDetector& detector, FaceRecognizer& recognizer) {
    std::cout << "\n=== 实时人脸检测 ===" << std::endl;
    std::cout << "按 'q' 退出, 按 's' 保存参考人脸" << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return;
    }
    
    std::vector<float> refFeature;
    bool hasReference = false;
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        auto faces = detector.detect(frame);
        
        for (size_t i = 0; i < faces.size(); i++) {
            float similarity = -1.0f;
            std::string label = "";
            
            if (hasReference && !faces.empty()) {
                auto feature = recognizer.extractFeature(frame, faces[i]);
                if (!feature.empty()) {
                    similarity = recognizer.compareFaces(refFeature, feature);
                    if (similarity > 0.6f) {
                        label = "Match";
                    } else {
                        label = "Unknown";
                    }
                }
            }
            
            drawFaceInfo(frame, faces[i], label, similarity);
        }
        
        // 显示信息
        std::string info = "Faces: " + std::to_string(faces.size());
        if (hasReference) {
            info += " | Reference set";
        }
        cv::putText(frame, info, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("实时人脸检测", frame);
        
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        } else if (key == 's' && !faces.empty()) {
            refFeature = recognizer.extractFeature(frame, faces[0]);
            hasReference = true;
            std::cout << "已保存参考人脸特征" << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    std::cout << "InsightFace C++ Demo - buffalo_sc 模型" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 模型路径
    std::string detectorModelPath = "models/det_10g.onnx";
    std::string recognizerModelPath = "models/w600k_r50.onnx";
    
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
        std::cout << "3. 实时检测: " << argv[0] << " webcam" << std::endl;
        std::cout << "\n示例:" << std::endl;
        std::cout << "  " << argv[0] << " detect test.jpg" << std::endl;
        std::cout << "  " << argv[0] << " compare person1.jpg person2.jpg" << std::endl;
        std::cout << "  " << argv[0] << " webcam" << std::endl;
        return 0;
    }
    
    std::string mode = argv[1];
    
    if (mode == "detect" && argc >= 3) {
        testDetection(detector, argv[2]);
    } else if (mode == "compare" && argc >= 4) {
        testRecognition(detector, recognizer, argv[2], argv[3]);
    } else if (mode == "webcam") {
        testWebcam(detector, recognizer);
    } else {
        std::cerr << "无效的命令或参数" << std::endl;
        return -1;
    }
    
    return 0;
}

