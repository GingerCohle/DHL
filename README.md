# This is the repo of DSGMNet, which contains the source code and visualization scripts.

## Install

### For 2080Ti (20+, 10+, Titan series) and Tesla V100 

#### conda create -n DSGMNet  python==3.7 -y

#### conda activate DSGMNet

#### conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch

#### git clone  https://github.com/GingerCohle/DSGMNet.git

#### cd DSGMNet

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

#### def _load_c2_pickled_weights(file_path):

```python
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
## Dataset



#### 
