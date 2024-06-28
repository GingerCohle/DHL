# This is the repo of DHL, which contains the source code and visualization scripts.

## 1, Installation

### For 2080Ti (20+, 10+, Titan series) and Tesla V100 

#### conda create -n DHL  python==3.7 -y

#### conda activate DHL

#### conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch

#### git clone  https://github.com/GingerCohle/DHL.git

#### cd DHL

#### git clone https://github.com/cocodataset/cocoapi.git

#### cd cocoapi/PythonAPI/

#### conda install ipython

#### pip install ninja yacs cython matplotlib tqdm 

#### pip install --no-deps torchvision==0.2.1

#### python setup.py build_ext install

#### cd../..

#### pip install opencv-python==3.4.17.63

#### pip install scikit-learn

#### pip install scikit-image

#### python setup.py build develop

#### pip install Pillow==7.1.0

#### pip install tensorflow tensorboardX

#### pip install ipdb

### For 3090 (30+) or 4090 (40+)

#### The PyTorch version should change to 1.9.1:

#### pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

#### Some of the codes need to be modified:

#### $\bullet$ fcos_core/utils/comm.py line 120: 

```python
def is_pytorch_1_1_0_or_later():
return True
```

#### $\bullet$ fcos_core/utils/c2_model_loading.py line 70: 

```python
def _load_c2_pickled_weights(file_path):
  with open(file_path, "rb") as f:
     if torch._six.PY37:
        data = pickle.load(f, encoding="latin1")
```

#### $\bullet$ fcos_core/utils/import.py:

```python
if torch._six.==PY37:== 4行
    import importlib
    import importlib.util
    import sys: 
```

## 2, Dataset

#### Street Sceen Dataset: Cityscape vs Foggy Cityscape, SIM10k vs Cityscape and KITTI vs Cityscape.

#### The file trees are as follow:

Quoted From SIGMA (CVPR2022 oral) [link](https://github.com/CityU-AIM-Group/SIGMA).

**Cityscapes -> Foggy Cityscapes**

- Download Cityscapes and Foggy Cityscapes dataset from the [link](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *leftImg8bit_trainvaltest.zip* for Cityscapes and *leftImg8bit_trainvaltest_foggy.zip* for Foggy Cityscapes.
- Download and extract the converted annotation from the following links: [Cityscapes and Foggy Cityscapes (COCO format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EZRXq3_5R_RKpAuTwjyYpWYBTjgKWZNuEjsgoYky31a96g?e=hfWAyl)

- Extract the training sets from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` to `Cityscapes/leftImg8bit/` directory.
- Extract the training and validation set from *leftImg8bit_trainvaltest_foggy.zip*, then move the folder `leftImg8bit_foggy/train/` and `leftImg8bit_foggy/val/` to `Cityscapes/leftImg8bit_foggy/` directory.

**Sim10k -> Cityscapes** (class car only)

- Download Sim10k dataset and Cityscapes dataset from the following links: [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *repro_10k_images.tgz* and *repro_10k_annotations.tgz* for Sim10k and *leftImg8bit_trainvaltest.zip* for Cityscapes.
- Download and extract the converted annotation from the following links: [Sim10k (VOC format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EQt48_9D1XtIiVE9GK3hFIYBQNOVSW4OfdZPtQAcCkS7bw?e=8NCweC) and [Cityscapes (COCO format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EZRXq3_5R_RKpAuTwjyYpWYBTjgKWZNuEjsgoYky31a96g?e=hfWAyl)

- Extract the training set from *repro_10k_images.tgz* and *repro_10k_annotations.tgz*, then move all images under `VOC2012/JPEGImages/` to `Sim10k/JPEGImages/` directory and move all annotations under `VOC2012/Annotations/` to `Sim10k/Annotations/`.
- Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.

**KITTI -> Cityscapes** (class car only)

- Download KITTI dataset and Cityscapes dataset from the following links: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *data_object_image_2.zip* for KITTI and *leftImg8bit_trainvaltest.zip* for Cityscapes.
- Download and extract the converted annotation from the following links: [KITTI (VOC format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EWqP3z9BpVNLlG3a_qGBO1EBNO7XO4GGaDlipixnlgc7rQ?e=LPBV5j) and [Cityscapes (COCO format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EZRXq3_5R_RKpAuTwjyYpWYBTjgKWZNuEjsgoYky31a96g?e=hfWAyl).
- Extract the training set from *data_object_image_2.zip*, then move all images under `training/image_2/` to `KITTI/JPEGImages/` directory.
- Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.

#### Before Training, you need to set the dataset path in file  [paths_catalog.py](https://github.com/CityU-AIM-Group/SIGMA/blob/main/fcos_core/config/paths_catalog.py). 

```python
DATA_DIR = {your data root}
```

#### The file trees are as follow:

#### [DATASET_PATH]

#### └─ Cityscapes

####    └── cocoAnnotations

####    └── leftImg8bit

####       └─── train

####       └─── val

####    └─ leftImg8bit_foggy

####       └── train

####       └── val

#### └─ KITTI

####    └── Annotations

####    └── ImageSets

####    └── JPEGImages

#### └─ Sim10k

####    └── Annotations

####    └── ImageSets

####    └── JPEGImages

#### PASCAL VOC vs COMIC and PASCAL VOC vs CLIPART

#### Coming soon!!!!

## 3, Training

Note that the training script in the model from the scratch with the default setting (batchsize = 4):

```bash
python tools/train_net_da.py \
        --config-file configs/SIGMA/xxx.yaml \
```

Test the well-trained model:

```bash
python tools/test_net.py \
        --config-file configs/SIGMA/xxx.yaml \
        MODEL.WEIGHT well_trained_models/xxx.pth
```



For example: test cityscapes to foggy cityscapes with VGG16 backbone.

```bash
python tools/test_net.py \
         --config-file configs/SIGMA/sigma_vgg16_cityscapace_to_foggy.yaml \
         MODEL.WEIGHT well_trained_models/city_to_foggy_vgg16_43.58_mAP.pth
```

## 3, Training

#### If you want to look up the main contribution code of DNet. 

```bash
./r50_sigma_city2foggy_iter45k.sh
```
#### If your gpu device is Tesla V100 (32G) and the virtual environment is the same as us. The training loss can be reproduce the same loss and ACC exactly [link](https://github.com/GingerCohle/DNet/blob/main/log/City2Foggy_45.5_log.txt).
