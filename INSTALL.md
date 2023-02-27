# Installation

SMC-Bench contains 4 task: Commonsense Reasoning, Arithmetic Reasoning, Multilingual Translation, and Protein Prediction. 

We provide a environment.yml which can create an environment that is compatible with four tasks, on an A100 GPU with CUDA 11.3. 
# To create an environment that is compatible with four tasks:
```commandline
conda env create --file=environment.yml
```

We also provide commands to specifically create environments for each task
# Commonsense Reasoning and Multilingual Translation 

Commonsense Reasoning and Multilingual Translation are highly relied on the implementation of [Fairseq](https://github.com/facebookresearch/fairseq) provied by Facebook.

## Requirements and Installation：

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* **To install fairseq** and develop locally:

``` bash
conda create -n SMC python=3.8 -y
conda activate SMC
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
git clone https://github.com/VITA-Group/SMC-Bench.git
cd SMC-Bench/Commonsense_Reasoning/
pip install --editable ./

```

# Arithmatic Reasoning (coming soon) 

Arithmetic Reasoning is implemented based on [SVAMP](https://github.com/arkilpatel/SVAMP).

