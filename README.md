# [ICLR 2023] Sparsity May Cry Benchmark (SMC-Bench)

Official PyTorch implementation of **SMC-Bench** - Sparsity May Cry: Let Us Fail (Current) Sparse Neural Networks Together!

[Shiwei Liu](https://shiweiliuiiiiiii.github.io/), [Tianlong Chen](https://tianlong-chen.github.io/about/), [Zhenyu Zhang](https://scholar.google.com/citations?user=ZLyJRxoAAAAJ&hl=zh-CN), [Xuxi Chen](http://xxchen.site/), [Tianjin Huang](https://research.tue.nl/en/persons/tianjin-huang), [Ajay Jaiswal](https://ajay1994.github.io/), [Zhangyang Wang](https://vita-group.github.io/)

University of Texas at Austin, Eindhoven University of Technology

The "Sparsity May Cry" Benchmark (SMC-Bench) is a collection of benchmark in pursuit of a more general evaluation and unveiling the true potential of sparse algorithms. SMC-Bench contains carefully curated 4 diverse tasks with 12 datasets, that accounts for capturing a wide-range of domain-specific knowledge. 


The benchmark organizers can be contacted at s.liu@tue.nl.

Table of contents
* [Installation](#installation-of-big-bench)

--- 
<p align="center">
<img src="https://github.com/VITA-Group/SMC-Bench/blob/main/Images/stay-tuned.png" width="600" height="350">
</p>

## Tasks, Models, and Datasets
Specifically, we consider a broad set of tasks including *commonsense reasoning, arithmatic reasoning, multilingual translation, and protein prediction*, whose content spans multiple domains, requiring a vast amount of commonsense knowledge, solid mathematical and scientific background to solve. Note that none of the datasets in SMC-Bench has been created from scratch for the benchmark, we rely on pre-existing datasets as they have been implicitly agreed by researchers as challenging, interesting, and of high practical value.  The models and datasets that we used for SMC-Bench are summarized below. 

--- 
<p align="center">
<img src="https://github.com/VITA-Group/SMC-Bench/blob/main/Images/Summary.png" width="800" height="350">
</p>

## Sparse Algorithms
*After Taining*: [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635), [Magnitude After Training](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf), [Random After Training](https://arxiv.org/abs/1812.10240), [oBERT](https://arxiv.org/abs/2203.07259).

*During Taining*: [Gradual Magnitude Pruning](https://arxiv.org/abs/1902.09574a).

*Before Training*: [Magnitude Before Training](https://arxiv.org/abs/2009.08576), [SNIP](https://arxiv.org/abs/1810.02340), [Rigging the Lottery](https://arxiv.org/abs/1911.11134), [Random Before Training](https://arxiv.org/abs/2202.02643).

## Evaluation

Commonsense Reasoning
--- 
<p align="center">
<img src="https://github.com/VITA-Group/SMC-Bench/blob/main/Images/Commonsense_reasoning.png" width="800" height="250">
</p>

Arithmatic Reasoning
--- 
<p align="center">
<img src="https://github.com/VITA-Group/SMC-Bench/blob/main/Images/Arithmatic_reasoning.png" width="800" height="500">
</p>

Protein Property Prediction
--- 
<p align="center">
<img src="https://github.com/VITA-Group/SMC-Bench/blob/main/Images/Protain_thermal_stability_prediction.png" width="800" height="250">
</p>

Multilingual Translation
--- 
<p align="center">
<img src="https://github.com/VITA-Group/SMC-Bench/blob/main/Images/Multilingual_translation.png" width="800" height="250">
</p>

## Installation of BIG-bench
