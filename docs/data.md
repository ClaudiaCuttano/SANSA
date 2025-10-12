# Data Preparation

Our setup follows [Matcher](https://github.com/aim-uofa/Matcher). <br>
Create a new directory `data` to store all the datasets.
At the end of the preparation, we expect the following directory structure:

```
SANSA/
├── data/
│   ├── COCO2014/           
│   │   ├── annotations/
│   │   │   ├── train2014/
│   │   │   └── val2014/
│   │   ├── train2014/
│   │   ├── val2014/
│   │   └── splits/
│   │       ├── trn/
│   │       └── val/
│   ├── FSS-1000/
│   │   ├── data/
│   │   │   ├── ab_wheel/
│   │   │   ├── ...
│   │   │   └── zucchini/
│   │   └── splits/
│   │       ├── trn.txt
│   │       ├── val.txt
│   │       └── test.txt
│   ├── LVIS/
│   │   ├── coco/
│   │   │   ├── train2017/
│   │   │   └── val2017/
│   │   ├── lvis_train.pkl
│   │   └── lvis_val.pkl
│   ├── PACO-Part/
│   │   ├── coco/
│   │   │   ├── train2017/
│   │   │   └── val2017/
│   │   ├── paco/
│   │   │   ├── paco_part_train.pkl
│   │   │   └── paco_part_val.pkl
│   ├── Pascal-Part/
│   │   ├── VOCdevkit/
│   │   │   ├── VOC2010/
│   │   │   │   ├── JPEGImages/
│   │   │   │   ├── Annotations_Part_json_merged_part_classes/
│   │   │   │   └── all_obj_part_to_image.json
```


### 🥥 COCO-20<sup>i</sup>

Download COCO2014 train/val images and annotations:
```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Download train/val annotations: [train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing), [val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing). <br>
Unzip and place both `train2014/` and `val2014/` under `data/COCO2014/annotations/`.

### 📸 FSS-1000
Download FSS-1000 images and annotations: [FSS-1000.zip](https://drive.google.com/file/d/1Fn-cUESMMF1pQy8Xff-vPQvXJdZoUlP3/view?usp=sharing).


### 🦉 LVIS-92<sup>i</sup>
Download COCO2017 train/val images: 
 ```
 wget http://images.cocodataset.org/zips/train2017.zip
 wget http://images.cocodataset.org/zips/val2017.zip
 ```
Download LVIS-92<sup>i</sup> extended mask annotations: [lvis.zip](https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view?usp=sharing).
Unzip and place the `.pkl` files under `data/LVIS/`.


### 🧩 PACO-Part
Download COCO2017 train/val images (same as LVIS): 
 ```
 wget http://images.cocodataset.org/zips/train2017.zip
 wget http://images.cocodataset.org/zips/val2017.zip
 ```
 Download PACO-Part extended mask annotations: [paco.zip](https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view?usp=sharing).
Unzip and place the `.pkl` files under `data/PACO-Part/paco/`.

### 🔩 Pascal-Part
 Download VOC2010 train/val images: 
 ```
 wget http://roozbehm.info/pascal-parts/trainval.tar.gz
 wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
 ```
 Download Pascal-Part extended mask annotations: [pascal.zip](https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view?usp=sharing).
Unzip everything into `data/Pascal-Part/VOCdevkit/VOC2010/`.

---------

## Optional datasets
We do not use these datasets in SANSA experiments,
but they are supported (`datasets/deepglobe.py`, `datasets/isic.py`, `datasets/lung.py`) and may be useful for others.

### 🌍 DeepGlobe
Download the dataset from the [DeepGlobe](https://drive.google.com/file/d/12Dljy04maKIim3mZsR50CEOC3_ROZLCg/view). 
Expected layout:

```
SANSA/
├── data/
│   ├── Deepglobe/     
│   │   ├── image/                                  
│   │   └── filter_mask/                            
```

### 🩺 ISIC 2018
Download the dataset from the [ISIC Challenge 2018](https://challenge.isic-archive.com/data/#2018).
Expected layout:

```
SANSA/
├── data/
│   ├── ISIC/     
│   │   ├── ISIC2018_Task1_Training_GroundTruth/
│   │   ├── ISIC2018_Task1-2_Training_Input/
```

### 🫁 Chest X-ray
Download the dataset from the [Chest X-ray](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels).
Expected layout:
```
SANSA/
├── data/
│   ├── LungSegmentation/ 
│   │   ├── CXR_png/
│   │   └── masks/
```