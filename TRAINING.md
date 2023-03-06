# Training

We provide training code, commands and requirements to enable running various sprase algorithms on SMC-Bench here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Commonsense Reasoning
We heavily rely on the awesome repo [fairseq](https://github.com/facebookresearch/fairseq) from FACEBOOK for Commonsense Reasoning. 

### CommonsenseQA 
The full instructions for CommonsenseQA can be found [here](Commonsense_Reasoning/examples/roberta/commonsense_qa/README.md).

More specifially, we need to
1. cd SMC-Bench/Commonsense_Reasoning/  
2. pip install --editable ./
3. Download CommonsenseQA dataset: bash examples/roberta/commonsense_qa/download_cqa_data.sh
4. Download roberta.large from [here](Commonsense_Reasoning/examples/roberta/README.md). 
5. Densely or sparsely finetuning with the commands provided [here](Commonsense_Reasoning/examples/roberta/commonsense_qa/README.md).

### WinoGrande 
The dense training on WinoGrande dataset can be found [here](https://github.com/VITA-Group/SMC-Bench/tree/main/Commonsense_Reasoning/examples/roberta/wsc#roberta-training-on-winogrande-dataset). To enable sparse training, or pruning, just follow the commands of [CommonsenseQA](Commonsense_Reasoning/examples/roberta/commonsense_qa/README.md).

### RACE 
The dense training on WinoGrande dataset can be found [here](https://github.com/VITA-Group/SMC-Bench/blob/main/Commonsense_Reasoning/examples/roberta/README.race.md). To enable sparse training, or pruning, just follow the commands of [CommonsenseQA](Commonsense_Reasoning/examples/roberta/commonsense_qa/README.md).
