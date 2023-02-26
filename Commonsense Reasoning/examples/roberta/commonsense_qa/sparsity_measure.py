import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
import os, re


import sys

save_path = '/home/sliu/project_space/pruning_fails/QA/robert/commonsenseqa'

removed_layers = ['in_proj_weight', 'out_proj_weight', 'fc1_weight', 'fc2_weight', 'lm_head.dense.weight']
snns = ['gm', 'gm_after',  'gmp', 'IMP', 'random', 'random_after',  'snip']

sparsity_IMP = ['checkpoint_best_iter1.pt','checkpoint_best_iter2.pt','checkpoint_best_iter3.pt','checkpoint_best_iter4.pt', \
                    'checkpoint_best_iter5.pt', 'checkpoint_best_iter6.pt', 'checkpoint_best_iter7.pt', 'checkpoint_best_iter8.pt', \
                    'checkpoint_best_iter9.pt', 'checkpoint_best_iter10.pt']
sparsities = ['0.2',  '0.36',  '0.488',  '0.590',  '0.672',  '0.738',  '0.791',  '0.8325' , '0.866' , '0.893']

sparsity_all = []
for snn in snns:
    snn_path = os.path.join(save_path, snn)

    if snn != 'IMP':
        for sparsity in sparsities:

            roberta = RobertaModel.from_pretrained(os.path.join(snn_path, sparsity), 'checkpoint_best.pt', 'data/CommonsenseQA')


            for name, weight in roberta.named_parameters():
                if len(weight.size()) == 2 or len(weight.size()) == 4:
                    if name in removed_layers: continue
                    sparsity_all.append((weight==0).sum().item() / weight.numel())
                    print(f'sparsity of {name} is {(weight == 0).sum().item() / weight.numel()}')
    else:
        for sparsity in sparsity_IMP:
            roberta = RobertaModel.from_pretrained(os.path.join(snn_path, '0.2'), str(sparsity),
                                                   'data/CommonsenseQA')
            for name, weight in roberta.named_parameters():
                if len(weight.size()) == 2 or len(weight.size()) == 4:
                    if name in removed_layers: continue
                    sparsity_all.append((weight == 0).sum().item() / weight.numel())
                    print(f'sparsity of {name} is {(weight == 0).sum().item() / weight.numel()}')

                

torch.save(sparsity_all, 'CSQA_sparsity.pt')




