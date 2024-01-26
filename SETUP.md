Certainly, here's a clear step-by-step markdown guide for setting up the environment on Eddie, including SSH access, Miniconda installation, and package dependencies:

````markdown
# Environment Setup Instructions for Eddie

Follow these instructions to set up your environment on the Eddie cluster.

## SSH into Eddie

```bash
ssh s2583684@eddie.ecdf.ed.ac.uk
```
````

## Navigate to the Scratch Space

Replace `s2583684` with your actual user ID.

```bash
cd /exports/eddie/scratch/s2583684/
```

## Install Miniconda

Download and install Miniconda in the scratch space.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

After the installation is complete, add Miniconda to the PATH in your `.bashrc` file.

```bash
echo ". /exports/eddie/scratch/s2583684/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
```

## Create a Conda Environment

Create a new Conda environment with Python 3.9.

```bash
mamba create -n myenv python=3.9
```

Activate the new environment.

```bash
conda activate myenv
```

## Install Dependencies

Install all the required dependencies within your `myenv`.

```bash
conda install -c conda-forge pycairo pygobject manimpango -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia -y
pip install transformers==4.17.0
pip install --upgrade pip
# Uncomment the following lines if applicable
# pip install /app/pixel/datasets
# pip install -e /app/pixel
pip install torch
pip install -r /app/requirements.txt
pip install --upgrade datasets
```

Make sure to replace `/app/requirements.txt` with the actual path to your `requirements.txt` file, and uncomment and adjust the paths in the `pip install` commands according to where your `datasets` and `pixel` modules are located.

After following these steps, your environment should be correctly set up to run your applications on Eddie.

```

Ensure that you replace placeholders such as `s2583684` with your actual user information and adjust the paths according to your specific directory structure and requirements.
```
