import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
import os, re

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

list = ['lr104/300/5epochs/', 'lr104/600/5epochs/', 'lr105/300/5epochs/', 'lr105/600/5epochs/', 'lr505/300/5epochs/', 'lr505/600/5epochs/', 'lr506/300/5epochs/', 'lr506/600/5epochs/',
               'lr104/300/10epochs/', 'lr104/600/10epochs/', 'lr105/300/10epochs/', 'lr105/600/10epochs/', 'lr505/300/10epochs/', 'lr505/600/10epochs/', 'lr506/300/10epochs/', 'lr506/600/10epochs/']

print(f'Evaluation list is {list}')

check_point_folder_source = '/home/sliu/project_space/pruning_fails/QA/robert/commonsenseqa/'

for method in ['obert_after_noemb_noclassifier_LRR/']:

    check_point_folder = check_point_folder_source + method
    # model_files = [0.5]
    model_files = os.listdir(check_point_folder)
    model_files = sorted_nicely(model_files)

    for file in model_files:
        print(file)
        roberta = RobertaModel.from_pretrained(os.path.join(check_point_folder, str(file)), 'checkpoint_best.pt', 'data/CommonsenseQA')

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