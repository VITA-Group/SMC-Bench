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

``` 
conda create -n SMC python=3.8 -y
conda activate SMC
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
git clone https://github.com/VITA-Group/SMC-Bench.git
cd SMC-Bench/Commonsense_Reasoning/
pip install --editable ./
pip install requests

```

# Arithmatic Reasoning  

Arithmetic Reasoning is implemented based on [SVAMP](https://github.com/arkilpatel/SVAMP). 

## Requirements and Installation：

```
cd SMC-Bench/Arithmetic_Reasoning/code/
pip install -r requirements.txt 
``` 
we might meet the following error  
```
Resource punkt not found.
Please use the NLTK Downloader to obtain the resource:
```
To solve this, run the commands below:
```
python
>>> import nltk
>>> nltk.download('punkt')
>>> quit()
```
If we encouter the no transpose attreibute error, i.e., "AttributeError: 'str' object has no attribute 'transpose'", we need to ajust the version of transformers. For instance, transformers==3.4.0 works for me. 
```
pip install  transformers==3.4.0 
```

# Multilingual Translation (Comming soon)

# Protein Prediction (Comming soon)
