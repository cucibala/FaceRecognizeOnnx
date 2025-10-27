import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Base64;

/**
 * InsightFace JNI 接口 Java 封装
 */
public class FaceRecognitionJNI {
    
    // 加载本地库
    static {
        System.loadLibrary("InsightFaceJNI"); // 实际库名根据编译结果调整
    }
    
    // 单个图片的Base64数据结构
    public static class ImageBase64 {
        public String base64Str;
        public int strLen;
        public Object userData;
        
        public ImageBase64(String base64Str) {
            this.base64Str = base64Str;
            this.strLen = base64Str.length();
            this.userData = null;
        }
    }
    
    // 单个处理结果
    public static class ImageResult {
        public float[] features;
        public int featureDim;
        public int status; // 0=成功，非0=错误码
        
        public boolean isSuccess() {
            return status == 0 && features != null && features.length > 0;
        }
    }
    
    // Native 方法声明
    private native int nativeInitialize(String modelPath);
    private native ImageResult[] nativeProcessBatch(String[] base64Images);
    private native float nativeCompareFaces(float[] features1, float[] features2);
    private native void nativeCleanup();
    
    // 实例变量
    private boolean initialized = false;
    
    /**
     * 初始化识别器
     * @param modelPath 模型文件路径
     * @return 0 表示成功，非 0 表示失败
     */
    public int initialize(String modelPath) {
        int result = nativeInitialize(modelPath);
        if (result == 0) {
            initialized = true;
        }
        return result;
    }
    
    /**
     * 批量处理 Base64 图片
     * @param base64Images Base64 编码的图片数组
     * @return 处理结果数组
     */
    public ImageResult[] processBatch(String[] base64Images) {
        if (!initialized) {
            throw new IllegalStateException("FaceRecognizer not initialized");
        }
        return nativeProcessBatch(base64Images);
    }
    
    /**
     * 比对两个特征向量
     * @param features1 第一个特征向量
     * @param features2 第二个特征向量
     * @return 相似度 [0, 1]，-1 表示错误
     */
    public float compareFaces(float[] features1, float[] features2) {
        if (!initialized) {
            throw new IllegalStateException("FaceRecognizer not initialized");
        }
        return nativeCompareFaces(features1, features2);
    }
    
    /**
     * 清理资源
     */
    public void cleanup() {
        if (initialized) {
            nativeCleanup();
            initialized = false;
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        cleanup();
        super.finalize();
    }
    
    // 工具方法：将图片文件转换为 Base64
    public static String imageFileToBase64(String imagePath) throws IOException {
        File file = new File(imagePath);
        FileInputStream fis = new FileInputStream(file);
        byte[] bytes = new byte[(int) file.length()];
        fis.read(bytes);
        fis.close();
        return Base64.getEncoder().encodeToString(bytes);
    }
    
    // 示例使用
    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage: java FaceRecognitionJNI <model_path> <image1> <image2>");
            return;
        }
        
        String modelPath = args[0];
        String image1Path = args[1];
        String image2Path = args[2];
        
        FaceRecognitionJNI fr = new FaceRecognitionJNI();
        
        try {
            // 1. 初始化
            System.out.println("Initializing...");
            int ret = fr.initialize(modelPath);
            if (ret != 0) {
                System.err.println("Initialization failed: " + ret);
                return;
            }
            
            // 2. 转换图片为 Base64
            System.out.println("Converting images to Base64...");
            String base64_1 = imageFileToBase64(image1Path);
            String base64_2 = imageFileToBase64(image2Path);
            
            String[] base64Images = {base64_1, base64_2};
            
            // 3. 批量处理
            System.out.println("Processing batch...");
            ImageResult[] results = fr.processBatch(base64Images);
            
            // 4. 查看结果
            System.out.println("\nResults:");
            for (int i = 0; i < results.length; i++) {
                System.out.println("Image " + (i + 1) + ":");
                if (results[i].isSuccess()) {
                    System.out.println("  Status: SUCCESS");
                    System.out.println("  Feature dimension: " + results[i].featureDim);
                    System.out.print("  First 10 features: ");
                    for (int j = 0; j < Math.min(10, results[i].features.length); j++) {
                        System.out.printf("%.4f ", results[i].features[j]);
                    }
                    System.out.println();
                } else {
                    System.out.println("  Status: FAILED (" + results[i].status + ")");
                }
            }
            
            // 5. 比对
            if (results[0].isSuccess() && results[1].isSuccess()) {
                System.out.println("\nComparing faces...");
                float similarity = fr.compareFaces(results[0].features, results[1].features);
                System.out.printf("Similarity: %.4f\n", similarity);
                
                float threshold = 0.6f;
                if (similarity > threshold) {
                    System.out.println("Result: Same person");
                } else {
                    System.out.println("Result: Different persons");
                }
            }
            
            // 6. 清理
            fr.cleanup();
            System.out.println("\nDone!");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

