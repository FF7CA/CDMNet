# CDMNet

**CDMNet** is an implementation of the paper CDMNet: A Cross-Domain Mamba Network with Joint Spatial-Frequency
Learning for Robust SAR Oil Spill Detection

**Note**：This repository is undergoing revisions, and our paper is still under review.

## Installation

```
conda create --n aerith python=3.12 && conda activate aerith
pip install -r requirement.txt
```

## Dataset

### Download the dataset

For SOS:

For M4D:

### Data processing

The dataset folder under */dataset* should as follows.

```
data
|----SOS
|--------train
|--------val
|----M4D
```

# Get start

## Training

```
python train_supervision.py -c ./config/SOS/CDMNet.py
```
```
python train_supervision.py -c ./config/M4D/CDMNet.py
```
## Testing

```
python SOS_seg_test.py -c ./config/SOS/CDMNet.py -o /root/results/SOS/CDMNet --rgb -t 'lr'
```
```
python M4D_test.py -c ./config/M4D/CDMNet.py -o /root/results/M4D/CDMNet --rgb -t 'lr'
```

## Thanks



