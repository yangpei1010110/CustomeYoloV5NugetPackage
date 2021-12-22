# CustomeYoloV5NugetPackage

# Usage Like This

// set labels
var labels = new string[]{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

// load custom model
var model = new YoloV5Model(modelFilePath, labels.Length + 5);

// read image as input data
var inputdata = new InputData() { Image = new Bitmap(Image.FromFile(filePath))};

// detect image and get result
var results = model.Detect(imageData);

var box = result.Box; // RectangleF
var label = labels[result.LabelIndex];
var confidence = result.Confidence;

// end
