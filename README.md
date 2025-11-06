Installation:
===
**By mamba/conda (recommanded)**

Ubuntu/Centos/Rocky/Windows:
---
```
mamba env create -f AI-HEAD.yaml

mamba activate AI_HEAD_IHRSR

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

pip install wxpython==4.2.1

pip install alive_progress
```
***The installation of wxPython 4.2.1 may failed, but that would not affect the usage.**

***If the installation of wxpython pip fails, you can try installing the gtk3.0 library on your system.**

Environments:
---
**1.You could simply get the environment file by:**
```
cd (where you install AI-HEAD)

chmod +x *.py

python create_env_file.py
```
**Then you will get a file called "AI_HEAD_ENV.sh",and a file called "IHRSR_ENV.sh"**


**2.Or you could make the environment file by yourself:**
```
export AI_HEAD_PATH=(where you install AI-HEAD)

export AI_HEAD_LIB_PATH=(where you install AI-HEAD)/utils

export PATH=(where you install AI-HEAD):$PATH
```
**3.If you want to use IHRSR, you could simply set the environment file like this:**
```
export PATH=(where you install AI-HEAD)/external/IHRSR_v1.5/:$PATH

export LD_LIBRARY_PATH=(where you install AI-HEAD)/external/libgfortran_for_ihrsr/:$LD_LIBRARY_PATH
```
Usage:
---
AI-HEAD provides a very friendly graphical interface. Once the operating environment is configured, you could open the interface and create new projects anywhere:

**Examples:**
```
source AI_HEAD_ENV.sh

source IHRSR_ENV.sh

AI-HEAD.py
```
**Additionally, we have launched AI-HEAD in commandline version:**
```
source AI_HEAD_ENV.sh

source IHRSR_ENV.sh

AI-HEAD_command.py --config config_files.txt
```
**When the AI-HEAD program is running, the configuration file corresponding to the command line version of the module is also generated.**

Both of them could get everything done, so just choose the version you like.

External_programs:
---
SPIDER program could be available at https://spider-em.github.io/SPIDER/docs/spi-download.html

IHRSR program could be got at /external




