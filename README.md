# LFANet: three-class segmentation using Pytorch.
This repository contains code for three-class segmentation model 


## Usage

#### Note : Use Python 3.8, Pytorch 1.4.1, CUDA 10.1

### Dataset
make sure to put the files as the following structure:
```
data
├── images
|   ├── 0a7e06.jpg
│   ├── 0aab0a.jpg
│   ├── 0b1761.jpg
│   ├── ...
|
└── masks
    ├── 0a7e06.png
    ├── 0aab0a.png
    ├── 0b1761.png
    ├── ...
```
mask is a single-channel category index. For example, your dataset has three categories, mask should be 8-bit images with value 0,1,2 as the categorical value, this image looks black.

### Training
```bash
python train.py
```

### inference
```base
python inference.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output
```

## Tensorboard
You can visualize in real time the train and val losses, along with the model predictions with tensorboard:
```bash
tensorboard --logdir=runs
```

