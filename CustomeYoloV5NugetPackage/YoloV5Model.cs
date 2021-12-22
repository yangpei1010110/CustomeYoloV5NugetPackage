using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Collections.Concurrent;
using System.Drawing;

namespace CustomeYoloV5NugetPackage
{
    /// <summary>
    /// 输入数据
    /// </summary>
    public class InputData
    {
        /// <summary>
        /// ColumnName 应该和 pipe 对应的 inputColumnName 相同
        /// </summary>
        [ColumnName("Image")]
        [ImageType(width: 640, height: 640)]
        public Bitmap Image { get; set; }
    }

    /// <summary>
    /// 输出数据
    /// </summary>
    public class OutputData
    {
        /// <summary>
        /// ColumnName 应该和 onnx 模型的输出数据名 相同
        /// </summary>
        [ColumnName("output")]
        public float[] Output { get; set; }
    }

    /// <summary>
    /// 结果数据
    /// </summary>
    public class ResultData
    {
        public RectangleF Box { get; set; }
        public float Confidence { get; set; }
        public int LabelIndex { get; set; }
        public ResultData(RectangleF box, float confidence, int labelIndex)
        {
            Box = box;
            Confidence = confidence;
            LabelIndex = labelIndex;
        }
    }

    public class YoloV5Model
    {
        /// <summary>
        /// 推断引擎
        /// </summary>
        public PredictionEngine<InputData, OutputData> PredictionEngine { get; private set; }
        /// <summary>
        /// 每条数据的尺寸 Dimensions = 模型可识别类别数量 + x + y + width + height + confidence
        /// </summary>
        public int Dimensions { get; private set; }
        /// <summary>
        /// 输出结果最大限制数量
        /// </summary>
        public int Limit { get; private set; }
        /// <summary>
        /// objectness 过滤参数
        /// </summary>
        public float Confidence { get; private set; }
        /// <summary>
        /// 类可信度过滤参数
        /// </summary>
        public float MulConfidence { get; private set; }
        /// <summary>
        /// 重复框过滤参数
        /// </summary>
        public float Overlap { get; private set; }
        /// <summary>
        /// 输入模型 width
        /// </summary>
        public int ModelWidth { get; private set; }
        /// <summary>
        /// 输入模型 height
        /// </summary>
        public int ModelHeight { get; private set; }
        /// <summary>
        /// 初始化推断类
        /// </summary>
        /// <param name="modelFilePath">yolo onnx 模型路径</param>
        /// <param name="dimensions">每条数据的尺寸</param>
        /// <param name="limit">输出结果最大限制数量</param>
        /// <param name="confidence">objectness 过滤参数</param>
        /// <param name="mulConfidence">类可信度过滤参数</param>
        /// <param name="overlap">重复框过滤参数</param>
        /// <param name="modelWidth">输入模型 width</param>
        /// <param name="modelHeight">输入模型 height</param>
        public YoloV5Model(string modelFilePath, int dimensions, 
            int limit = 10,float confidence = 0.2f, float mulConfidence = 0.25f, float overlap = 0.45f, int modelWidth = 640, int modelHeight = 640)
        {
            Dimensions = dimensions;
            Limit = limit;
            Confidence = confidence;
            MulConfidence = mulConfidence;
            Overlap = overlap;
            ModelWidth = modelWidth;
            ModelHeight = modelHeight;

            MLContext mlContext = new MLContext();
            var transformer = mlContext.Transforms
                .ResizeImages(outputColumnName: "images", imageWidth: modelWidth, imageHeight: modelHeight, inputColumnName: "Image", resizing: ImageResizingEstimator.ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels("images", scaleImage: 1f / 255f, interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                        gpuDeviceId: 0, // 显卡 index
                        shapeDictionary: new Dictionary<string, int[]>()
                                    {
                                       { "images", new[] { 1, 3, modelWidth, modelHeight } }
                                    },
                        outputColumnName: "output",
                        inputColumnName: "images",
                        modelFile: modelFilePath))
                .Fit(mlContext.Data.LoadFromEnumerable(new List<InputData>()));

            PredictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(transformer);
        }

        /// <summary>
        /// 推断并返回结果
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public IEnumerable<ResultData> Detect(InputData inputData)
        {
            var output = PredictionEngine.Predict(inputData).Output;
            var originData = OutputToResultData(output, inputData.Image.Width, inputData.Image.Height);
            var resultData = NonMaximumSuppression(originData);

            return resultData;
        }

        /// <summary>
        /// 输出结果解析
        /// </summary>
        /// <param name="output"></param>
        /// <param name="imageWidth"></param>
        /// <param name="imageHeight"></param>
        /// <returns></returns>
        private IEnumerable<ResultData> OutputToResultData(float[] output,int imageWidth,int imageHeight)
        {
            // 计算图片大小与模型大小的倍率
            var (xGain, yGain) = ((float)ModelWidth / (float)imageWidth, (float)ModelHeight / (float)imageHeight);
            var gain = Math.Min(xGain, yGain);
            // left, right pads
            var (xPad, yPad) = ((ModelWidth - imageWidth * gain) / 2, (ModelHeight - imageHeight * gain) / 2);

            var result = new ConcurrentBag<ResultData>();
            Parallel.For(0, output.Length / Dimensions, (index) =>
            {
                var currentIndex = index * Dimensions;
                // 过滤 objectness 过低的数据
                if (output[currentIndex + 4] < Confidence)
                {
                    return;
                }
                // 为每个类的 confidence 乘以 objectness
                Parallel.For(5, Dimensions, (classIndex) =>
                {
                    output[currentIndex + classIndex] = output[currentIndex + classIndex] * output[currentIndex + 4];
                });

                // 收集对象
                Parallel.For(5, Dimensions, (classIndex) =>
                {
                    // 过滤可信度低的对象
                    if (output[currentIndex + classIndex] < MulConfidence)
                    {
                        return;
                    }
                    // 坐标转换
                    float topLeftX = ((output[currentIndex] - output[currentIndex + 2] / 2f) - xPad) / gain;
                    float topLeftY = ((output[currentIndex + 1] - output[currentIndex + 3] / 2f) - yPad) / gain;
                    float bottomRightX = ((output[currentIndex] + output[currentIndex + 2] / 2f) - xPad) / gain;
                    float bottomRightY = ((output[currentIndex + 1] + output[currentIndex + 3] / 2f) - yPad) / gain;
                    // 收集对象
                    result.Add(new ResultData(new RectangleF(topLeftX, topLeftY, bottomRightX - topLeftX, bottomRightY - topLeftY),
                        output[index * Dimensions + classIndex],
                        classIndex - 5));
                });
            });

            return result;
        }
        /// <summary>
        /// NMS 算法去除重复框
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        private IEnumerable<ResultData> NonMaximumSuppression(IEnumerable<ResultData> source)
        {
            // 根据可信度降序排列
            var sortSource = source.Select((resultData, index) => new { Box = resultData, Index = index })
                .OrderByDescending(b => b.Box.Confidence)
                .ToArray();

            var activeCount = sortSource.Count();
            var isActive = Enumerable.Repeat(true, activeCount).ToArray();

            // 准备结果
            var result = new List<ResultData>();

            for (int currentIndex = 0; currentIndex < sortSource.Count(); currentIndex++)
            {
                // 当前项可用
                if (isActive[currentIndex])
                {
                    var currentBox = sortSource[currentIndex].Box;
                    result.Add(currentBox);
                    // 检查结果数量 超过就返回
                    if (result.Count() > Limit)
                    {
                        break;
                    }

                    // 准备检查相邻边界
                    for (int checkIndex = currentIndex + 1; checkIndex < sortSource.Count(); checkIndex++)
                    {
                        // 当前项可用
                        if (isActive[checkIndex])
                        {
                            var checkBox = sortSource[checkIndex].Box;
                            // 检查重叠
                            if (Overlap < IntersectionOverUnion(currentBox.Box, checkBox.Box))
                            {
                                // 如果重叠区域超过阈值
                                isActive[checkIndex] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// IoU算法 求空间占比
        /// </summary>
        /// <returns>IoU</returns>
        private static float IntersectionOverUnion(RectangleF boxA, RectangleF boxB)
        {
            var areaA = boxA.Width * boxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boxB.Width * boxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boxA.Left, boxB.Left);
            var minY = Math.Max(boxA.Top, boxB.Top);
            var maxX = Math.Min(boxA.Right, boxB.Right);
            var maxY = Math.Min(boxA.Bottom, boxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);
            var unionArea = (areaA + areaB - intersectionArea);

            return intersectionArea / unionArea;
        }
    }
}
