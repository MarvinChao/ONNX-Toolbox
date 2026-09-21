# Test Model Download

The toolbox includes an on-demand downloader for a curated set of **101 popular
CNN / RNN ONNX models** (no LLMs). It is meant for pulling a local corpus of
real models to exercise the analysis pipeline against actual graphs.

> **Where the code lives:** the downloader is
> [`utils/test_model_download.py`](../utils/test_model_download.py). This
> document (in `docs/`) is reference material only. Models are saved to the
> `models/` directory at the repository root, which is git-ignored so large
> downloads are never committed. Run all commands from the repository root.

## Source

All models come from the community **ONNX Model Zoo** mirror hosted on
Hugging Face ([onnxmodelzoo](https://huggingface.co/onnxmodelzoo)). Each model
is fetched from:

```
https://huggingface.co/onnxmodelzoo/<slug>/resolve/main/<slug>.onnx
```

These are the models originally published in the
[onnx/models](https://github.com/onnx/models) repository. Please refer to that
repository and the individual Hugging Face model cards for licensing and
attribution of each model.

## Usage

```bash
# List the catalog without downloading anything
python utils/test_model_download.py --list

# Download everything (large: several GB total)
python utils/test_model_download.py --all

# Download only certain categories
python utils/test_model_download.py --category classification detection

# Download specific models by slug
python utils/test_model_download.py --models resnet50-v1-7 mobilenetv2-12

# Skip anything larger than 60 MB
python utils/test_model_download.py --all --max-mb 60

# Re-download even if the file already exists
python utils/test_model_download.py --models mnist-12 --force

# Print each model's input/output shapes after downloading
python utils/test_model_download.py --models mnist-12 --describe
```

Existing valid files are skipped by default, so re-running resumes a partial
corpus. Each downloaded file is validated with `onnx.checker`.

## Catalog

101 models across 6 categories. File sizes below are the exact bytes reported
by the server, shown rounded. Input/output shapes use ONNX dimension notation;
`N` / `batch_size` denotes a dynamic batch dimension. Classification models are
ImageNet-trained (1000 classes) unless noted.

> Shapes were taken from the ONNX Model Zoo model specifications; a
> representative subset (MNIST, ResNet/MobileNet/SqueezeNet/VGG families,
> super-resolution, version-RFB) was verified directly by loading the model and
> reading its graph. Within a family the input/output signature is identical
> across opset/quantization variants; only the file size differs.

### Classification (64)

Standard ImageNet signature unless stated: input `[N, 3, 224, 224]` (float),
output `[N, 1000]` class scores.

| Model (slug) | Size | Input shape | Output shape |
|---|---|---|---|
| resnet18-v1-7 | 46.8 MB | data: [N, 3, 224, 224] | resnetv1x_dense0_fwd: [N, 1000] |
| resnet34-v1-7 | 87.3 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet50-v1-7 | 102.6 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet101-v1-7 | 178.9 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet152-v1-7 | 241.8 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet18-v2-7 | 46.8 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet50-v2-7 | 102.4 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet34-v2-7 | 87.3 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet101-v2-7 | 178.7 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet152-v2-7 | 241.5 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet50-v1-12 | 102.6 MB | data: [N, 3, 224, 224] | [N, 1000] |
| resnet50-caffe2-v1-9 | 102.5 MB | gpu_0/data_0: [N, 3, 224, 224] | gpu_0/softmax_1: [N, 1000] |
| mobilenetv2-7 | 14.2 MB | data: [N, 3, 224, 224] | mobilenetv20_output_flatten0_reshape0: [N, 1000] |
| mobilenetv2-10 | 14.0 MB | input: [N, 3, 224, 224] | output: [N, 1000] |
| mobilenetv2-12 | 14.0 MB | input: [N, 3, 224, 224] | output: [N, 1000] |
| mobilenetv2-12-int8 | 3.7 MB | input: [N, 3, 224, 224] | output: [N, 1000] |
| mobilenetv2-12-qdq | 3.6 MB | input: [N, 3, 224, 224] | output: [N, 1000] |
| squeezenet1.0-7 | 5.0 MB | data_0: [1, 3, 224, 224] | softmaxout_1: [1, 1000] |
| squeezenet1.1-7 | 5.0 MB | data: [N, 3, 224, 224] | squeezenet0_flatten0_reshape0: [N, 1000] |
| squeezenet1.0-9 | 5.0 MB | data_0: [1, 3, 224, 224] | softmaxout_1: [1, 1000] |
| squeezenet1.0-12 | 5.0 MB | data_0: [1, 3, 224, 224] | softmaxout_1: [1, 1000] |
| vgg16-7 | 553.4 MB | data: [N, 3, 224, 224] | vgg0_dense2_fwd: [N, 1000] |
| vgg16-12 | 553.4 MB | data: [N, 3, 224, 224] | [N, 1000] |
| vgg16-bn-7 | 553.5 MB | data: [N, 3, 224, 224] | [N, 1000] |
| vgg19-7 | 574.7 MB | data: [N, 3, 224, 224] | [N, 1000] |
| vgg19-bn-7 | 574.8 MB | data: [N, 3, 224, 224] | [N, 1000] |
| vgg19-caffe2-9 | 574.7 MB | data_0: [N, 3, 224, 224] | prob_1: [N, 1000] |
| bvlcalexnet-3 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| bvlcalexnet-7 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| bvlcalexnet-9 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| bvlcalexnet-12 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| caffenet-3 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| caffenet-7 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| caffenet-9 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| caffenet-12 | 243.9 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| googlenet-3 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| googlenet-7 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| googlenet-9 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| googlenet-12 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| googlenet-12-int8 | 7.1 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| inception-v1-7 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| inception-v1-9 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| inception-v1-12 | 28.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| inception-v1-12-int8 | 10.2 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| inception-v2-7 | 45.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| inception-v2-9 | 45.0 MB | data_0: [1, 3, 224, 224] | prob_1: [1, 1000] |
| zfnet512-7 | 349.0 MB | gpu_0/data_0: [1, 3, 224, 224] | gpu_0/softmax_1: [1, 1000] |
| zfnet512-9 | 349.0 MB | gpu_0/data_0: [1, 3, 224, 224] | gpu_0/softmax_1: [1, 1000] |
| zfnet512-12 | 349.0 MB | gpu_0/data_0: [1, 3, 224, 224] | gpu_0/softmax_1: [1, 1000] |
| densenet-7 | 32.7 MB | data_0: [1, 3, 224, 224] | fc6_1: [1, 1000, 1, 1] |
| densenet-9 | 32.7 MB | data_0: [1, 3, 224, 224] | fc6_1: [1, 1000, 1, 1] |
| densenet-12 | 32.7 MB | data_0: [1, 3, 224, 224] | fc6_1: [1, 1000, 1, 1] |
| shufflenet-9 | 5.7 MB | gpu_0/data_0: [1, 3, 224, 224] | gpu_0/softmax_1: [1, 1000] |
| shufflenet-v2-10 | 9.2 MB | input: [N, 3, 224, 224] | output: [N, 1000] |
| shufflenet-v2-12 | 9.2 MB | input: [N, 3, 224, 224] | output: [N, 1000] |
| rcnn-ilsvrc13-7 | 230.8 MB | data_0: [1, 3, 224, 224] | fc-rcnn_1: [1, 200] |
| rcnn-ilsvrc13-9 | 230.8 MB | data_0: [1, 3, 224, 224] | fc-rcnn_1: [1, 200] |
| efficientnet-lite4-11 | 51.9 MB | images:0: [N, 224, 224, 3] | Softmax:0: [N, 1000] |
| efficientnet-lite4-11-int8 | 13.6 MB | images_quant:0: [N, 224, 224, 3] | Softmax:0: [N, 1000] |
| efficientnet-lite4-11-qdq | 13.5 MB | images:0: [N, 224, 224, 3] | Softmax:0: [N, 1000] |
| mnist-1 | 27.3 KB | Input3: [1, 1, 28, 28] | Plus214_Output_0: [1, 10] |
| mnist-7 | 26.5 KB | Input3: [1, 1, 28, 28] | Plus214_Output_0: [1, 10] |
| mnist-8 | 26.5 KB | Input3: [1, 1, 28, 28] | Plus214_Output_0: [1, 10] |
| mnist-12 | 26.1 KB | Input3: [1, 1, 28, 28] | Plus214_Output_0: [1, 10] |

Notes:
- `efficientnet-lite4` uses NHWC input `[N, 224, 224, 3]` (TensorFlow layout).
- `densenet` outputs `[1, 1000, 1, 1]` (unpooled).
- `rcnn-ilsvrc13` classifies into 200 ILSVRC2013 detection classes.
- `mnist` uses grayscale `[1, 1, 28, 28]` and outputs 10 digit logits.

### Detection (19)

| Model (slug) | Size | Input shape | Output shape(s) |
|---|---|---|---|
| ssd-10 | 80.4 MB | image: [N, 3, 1200, 1200] | bboxes, labels, scores |
| ssd-12 | 80.4 MB | image: [N, 3, 1200, 1200] | bboxes, labels, scores |
| ssd_mobilenet_v1_10 | 29.3 MB | inputs: [N, H, W, 3] (uint8) | detection boxes/classes/scores/num |
| ssd_mobilenet_v1_12 | 29.5 MB | inputs: [N, H, W, 3] (uint8) | detection boxes/classes/scores/num |
| ssd_mobilenet_v1_12-int8 | 9.5 MB | inputs: [N, H, W, 3] (uint8) | detection boxes/classes/scores/num |
| tiny-yolov3-11 | 35.5 MB | input_1: [N, 3, H, W]; image_shape: [N, 2] | boxes, scores, indices |
| yolov3-10 | 247.9 MB | input_1: [N, 3, H, W]; image_shape: [N, 2] | boxes, scores, indices |
| yolov3-12 | 247.9 MB | input_1: [N, 3, H, W]; image_shape: [N, 2] | boxes, scores, indices |
| yolov4 | 257.5 MB | input_1:0: [1, 416, 416, 3] | 3 feature maps (grids × anchors) |
| tinyyolov2-7 | 63.5 MB | image: [1, 3, 416, 416] | grid: [1, 125, 13, 13] |
| tinyyolov2-8 | 63.5 MB | image: [1, 3, 416, 416] | grid: [1, 125, 13, 13] |
| yolov2-coco-9 | 203.9 MB | input.1: [1, 3, 416, 416] | [1, 425, 13, 13] |
| FasterRCNN-10 | 167.3 MB | image: [3, H, W] | boxes, labels, scores |
| FasterRCNN-12 | 176.7 MB | image: [3, H, W] | boxes, labels, scores |
| FasterRCNN-12-int8 | 44.6 MB | image: [3, H, W] | boxes, labels, scores |
| MaskRCNN-10 | 177.9 MB | image: [3, H, W] | boxes, labels, scores, masks |
| MaskRCNN-12 | 178.0 MB | image: [3, H, W] | boxes, labels, scores, masks |
| MaskRCNN-12-int8 | 45.8 MB | image: [3, H, W] | boxes, labels, scores, masks |
| retinanet-9 | 228.4 MB | input: [1, 3, 480, 640] | boxes, scores, labels |

Notes:
- SSD / YOLOv3 / Faster/Mask R-CNN accept dynamic spatial dimensions (`H`, `W`).
- YOLO models output raw grid tensors that require post-processing (anchors,
  NMS); Faster/Mask R-CNN emit decoded boxes/labels/scores (and masks).

### Segmentation (3)

| Model (slug) | Size | Input shape | Output shape |
|---|---|---|---|
| fcn-resnet50-11 | 141.2 MB | input: [N, 3, H, W] | out: [N, 21, H, W] |
| fcn-resnet50-12 | 141.2 MB | input: [N, 3, H, W] | out: [N, 21, H, W] |
| fcn-resnet101-11 | 217.1 MB | input: [N, 3, H, W] | out: [N, 21, H, W] |

Notes:
- Fully-convolutional segmentation over 21 Pascal VOC classes; per-pixel output.

### Face & Body (9)

| Model (slug) | Size | Input shape | Output shape(s) |
|---|---|---|---|
| arcfaceresnet100-8 | 261.0 MB | data: [N, 3, 112, 112] | fc1: [N, 512] embedding |
| version-RFB-320 | 1.3 MB | input: [1, 3, 240, 320] | scores: [1, 4420, 2]; boxes: [1, 4420, 4] |
| version-RFB-320-int8 | 458 KB | input: [1, 3, 240, 320] | scores: [1, 4420, 2]; boxes: [1, 4420, 4] |
| version-RFB-640 | 1.6 MB | input: [1, 3, 480, 640] | scores, boxes |
| age_googlenet | 24.0 MB | input: [1, 3, 224, 224] | loss3/loss3_Y: [1, 8] age buckets |
| gender_googlenet | 23.9 MB | input: [1, 3, 224, 224] | loss3/loss3_Y: [1, 2] |
| emotion-ferplus-2 | 35.0 MB | Input3: [1, 1, 64, 64] | Plus692_Output_0: [1, 8] |
| emotion-ferplus-7 | 35.0 MB | Input3: [1, 1, 64, 64] | Plus692_Output_0: [1, 8] |
| emotion-ferplus-8 | 35.0 MB | Input3: [1, 1, 64, 64] | Plus692_Output_0: [1, 8] |

Notes:
- `arcface` produces a 512-D face embedding (not class scores).
- `version-RFB` is the Ultra-Light-Fast face detector; `-320`/`-640` differ in
  input resolution and anchor count.
- `emotion-ferplus` takes grayscale `64×64` faces, outputs 8 emotion scores.

### Style Transfer (5)

All fast-neural-style models share the same signature; only the trained style
differs. Input/output are dynamic-resolution RGB images.

| Model (slug) | Size | Input shape | Output shape |
|---|---|---|---|
| candy-9 | 6.7 MB | input1: [N, 3, H, W] | output1: [N, 3, H, W] |
| mosaic-9 | 6.7 MB | input1: [N, 3, H, W] | output1: [N, 3, H, W] |
| pointilism-9 | 6.7 MB | input1: [N, 3, H, W] | output1: [N, 3, H, W] |
| rain-princess-9 | 6.7 MB | input1: [N, 3, H, W] | output1: [N, 3, H, W] |
| udnie-9 | 6.7 MB | input1: [N, 3, H, W] | output1: [N, 3, H, W] |

### Super Resolution (1)

| Model (slug) | Size | Input shape | Output shape |
|---|---|---|---|
| super-resolution-10 | 234.5 KB | input: [batch_size, 1, 224, 224] | output: [batch_size, 1, 672, 672] |

Notes:
- Sub-pixel CNN that upscales the luminance (Y) channel 3× (224 → 672).

## Total size

Downloading the **entire** catalog is roughly **9-10 GB**. Use `--category`,
`--models`, or `--max-mb` to fetch a smaller working set. For a quick, tiny
corpus (a few MB total), the `mnist-*`, `super-resolution-10`,
`version-RFB-320`, and style-transfer models are good choices.
