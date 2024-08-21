# SAM2_fine_tune
The Segment Anything Model 2 (SAM 2) is an advanced foundational model designed to tackle prompt-based visual segmentation in both images and videos. 

## Description
The model utilizes a straightforward transformer architecture combined with streaming memory for efficient processing. SAM 2, trained on our customized dataset, delivers strong performance by using fine-tuning strategies that use parameter-efficient learning in both the encoder and decoder are superior to other strategies.

## Getting Started

### Dependencies
* GDAL, Pytorch- rasterio ... (see installation)
* Cuda-capable GPU [overview here](https://developer.nvidia.com/cuda-gpus)
* Anaconda [download here](https://www.anaconda.com/download)
* developed on Windows 10

### nstallation
* conda create -n sam2 python=3.11
* conda activate sam2
* pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
* cd ../SAM2_fine_tune/environment
* pip install -r requirements.txt

### Executing program
set parameters and run in run_pipeline.py

## Authors

* [Benjamin Stöckigt](https://github.com/benjaminstoeckigt)
* [Shadi Ghantous](https://github.com/Shadiouss)

## Acknowledgments

[segment-anything-2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints)
[Train/Fine-Tune Segment Anything 2 (SAM 2) in 60 lines of code](https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3)


