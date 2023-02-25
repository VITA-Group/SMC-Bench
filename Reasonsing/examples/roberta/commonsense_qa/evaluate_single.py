import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
import os, re


import sys


roberta = RobertaModel.from_pretrained(sys.argv[1], 'checkpoint_best.pt', 'data/CommonsenseQA')

total_zero = 0
total_weight = 0
for name, weight in roberta.named_parameters():
    total_zero += (weight==0).sum().item()
    total_weight += weight.numel()

print(f'the sparsity level of the model is {total_zero/total_weight} ')

roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0

with open('/home/sliu/Projects/fairseq/data/CommonsenseQA/valid.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            input = roberta.encode(
                'Q: ' + example['question']['stem'],
                'A: ' + choice['text'],
                no_separator=True
            )
            score = roberta.predict('sentence_classification_head', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

print('Accuracy: ' + str(ncorrect / float(nsamples)))