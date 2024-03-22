# Environment Setup Instructions for Eddie

Follow these instructions to set up your environment on the Eddie cluster.

## SSH into Eddie

```bash
ssh <student_id>@eddie.ecdf.ed.ac.uk
```

## Navigate to the Scratch Space

Replace `<student_id>` with your actual user ID.

```bash
cd /exports/eddie/scratch/<student_id>/
```

## Install Miniconda

Download and install Miniconda in the scratch space.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

After the installation is complete, add Miniconda to the PATH in your `.bashrc` file.

```bash
echo ". /exports/eddie/scratch/<student_id>/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
```

## Create a Conda Environment

Create a new Conda environment with Python 3.9.

```bash
conda create -n myenv python=3.9 --silent
```

Activate the new environment.

```bash
conda activate myenv
```

## Install Dependencies

Install all the required dependencies within your `myenv`.

- If you get memory error then request a CPU by using `qlogin -l h_vmem=4G`. you'll have to again move to scratch directory and activate the environment as well.

```bash
conda install -c conda-forge pycairo pygobject manimpango -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y
pip install transformers==4.17.0
pip install --upgrade pip
pip install torch
pip install -r /app/requirements.txt
pip install --upgrade datasets
```

cd into the main pixel directory and install the submodules

```bash

pip install ./datasets
```

After following these steps, your environment should be correctly set up to run your applications on Eddie.

```

Ensure that you replace placeholders such as `<student_id>` with your actual user information and adjust the paths according to your specific directory structure and requirements.
```

## Mamba installation

When using the mamba package you need to have a GPU to use. Make sure to be on a GPU when installing the library. On Eddie this involves activating the toolkit using the following commands

```bash
module avail cuda
module load cuda/12.1.1
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
git clone [causal1d lib]
cd causal1d lib
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .


```

## Litmus test to check if everything has worked

Request a GPU interactive session with

```bash
qlogin -q gpu -pe gpu-a100 2 -l h_vmem=500G -l h_rt=24:00:00
```

Remember to Activate your conda env (sometimes it will deactivate)

To Download models and dataset - add your hugging face token in "use_auth_token" within configs/training/pretraining.json
and also, set "dataset_caches" with location as string where you want hugging face to download the caches

navigate to the Edi-pixel directory and run the following

1. To start pre-training:
```bash
 python scripts/training/run_pretraining.py configs/training/pretraining.json
```

2. To start fine-tuning:
```bash
python scripts/training/run_glue_pixba.py configs/finetuning/glue.json
```

