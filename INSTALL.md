SMC-Bench contains 4 task: Commonsense Reasoning, Arithmetic Reasoning, Multilingual Translation, and Protein Prediction. 

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

# Arithmatic Reasoning 

Arithmetic Reasoning is implemented based on the 
