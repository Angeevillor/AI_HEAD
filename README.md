Installation:
===
**By mamba/conda (recommanded)**

Ubuntu:
---
```
mamba env create -f AI-HEAD.yaml

mamba activate AI_HEAD_IHRSR

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

pip install wxpython==4.2.1
```

Centos/Rocky:
---
```
mamba env create -f AI-HEAD.yaml

mamba activate AI_HEAD_IHRSR

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```
***The installation of wxPython 4.2.1 may failed in Centos/Rocky, but that would not affect the usage.**


Environments:
---
**1.You could simply just get the environment file by:**
```
cd (where you install AI-HEAD)

python create_env_file.py

Then you will get a file called "AI_HEAD_ENV.sh"
```
**2.Or you could make the environment file by yourself:**
```
export AI_HEAD_PATH=(where you install AI-HEAD)

export AI_HEAD_LIB_PATH=(where you install AI-HEAD)/utils

export PATH=(where you install AI-HEAD):$PATH
```
Usage:
---
```
source AI_HEAD_ENV.sh

AI-HEAD.py
```
