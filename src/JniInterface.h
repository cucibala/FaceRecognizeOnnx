// 定义单个图片的Base64数据结构
typedef struct {
    const char* base64_str;  // Base64字符串（Java传入，C++不负责释放）
    int str_len;             // 字符串长度（避免strlen重复计算）
} ImageBase64;

// 定义批量输入
typedef struct {
    ImageBase64* images;     // 图片数组
    int count;               // 图片数量
} BatchImageInput;

// 定义单个处理结果（示例：特征向量）
typedef struct {
    float* features;         // 特征数据（C++分配，Java需调用释放函数）
    int feature_dim;         // 特征维度
    int status;              // 处理状态（0=成功，非0=错误码）
} ImageResult;

// 定义批量输出
typedef struct {
    ImageResult* results;    // 结果数组
    int count;               // 结果数量（应与输入count一致）
} BatchImageOutput;

// 核心处理接口：输入批量Base64，输出处理结果
// 返回值：0=成功，非0=错误码（如参数无效、内存不足）
extern "C" int FR_ProcessBatchImages(
    const BatchImageInput* input,
    BatchImageOutput* output  // 输出结果，需提前分配results数组内存
);

// 结果释放接口：由Java调用，释放C++分配的内存
extern "C" void FR_FreeBatchResults(BatchImageOutput* output);

// 初始化接口：加载模型（可选，也可在第一次调用时自动加载）
extern "C" int FR_Initialize(const char* model_path);

// 初始化接口（带 GPU 选项）：use_gpu=true 启用 GPU，device_id 指定设备
extern "C" int FR_InitializeWithGPU(const char* model_path, bool use_gpu, int device_id);

// 清理接口：释放识别器资源
extern "C" void FR_Cleanup();

// 特征比对接口：计算两个特征向量的相似度
// 返回值：[0, 1] 范围的相似度，-1 表示错误
extern "C" float FR_CompareFaces(
    const float* features1, int dim1,
    const float* features2, int dim2
);