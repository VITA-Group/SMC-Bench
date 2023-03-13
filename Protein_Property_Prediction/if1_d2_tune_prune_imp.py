import numpy as np
import esm
import pickle
import torch
import torch.nn as nn
import sys
from esm.modules import TransformerLayer
from torch.nn.utils import prune
split_num = sys.argv[1]
split = pickle.load(open(f"/home/xc4863/clean_datasets/d2/d2_{split_num}_classification.pkl", "rb"))
backbone_lr = float(sys.argv[3])
lr = float(sys.argv[2])
epoch = int(sys.argv[4])
pruning_ratio=float(sys.argv[5])
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.cuda()
linear = nn.Sequential( nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 5)).cuda() 
optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=5e-2)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=int(epoch))
backbone_optimizer = torch.optim.AdamW(model.parameters(), lr=backbone_lr, weight_decay=5e-2)
checkpoint = torch.load(f"dense_d2_esm1f_{split_num}.pth.tar")
model.load_state_dict(checkpoint['model'])
linear.load_state_dict(checkpoint['linear'])
best_acc = 0

def pruning_model(model, px, method='omp'):
    

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        if 'decoder' in name:
            continue
        if 'self_attn' in name and isinstance(m, nn.Linear):
            print(f"Pruning {name}")
            parameters_to_prune.append((m,'weight'))
        if isinstance(m, TransformerLayer):
            print(f"Pruning {name}.fc1")
            parameters_to_prune.append((m.fc1,'weight'))
            print(f"Pruning {name}.fc2")
            parameters_to_prune.append((m.fc2,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    if method == 'omp':
        prune_method = prune.L1Unstructured
    elif method == 'rp':
        prune_method = prune.RandomUnstructured
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune_method,
        amount=px,
    )

for _round in range(9):
    best_acc = 0 
    optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=5e-2)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=int(epoch))
    backbone_optimizer = torch.optim.AdamW(model.parameters(), lr=backbone_lr, weight_decay=5e-2)
    pruning_model(model, pruning_ratio)
    
    for epoch_ in range(epoch):
        outputs = []
        labels = []
        for name, label in zip(split['train_names'], split['train_labels']):
            fpath = f"/home/xc4863/clean_datasets/d2/d2_clean/{name}/unrelaxed_model_1_ptm.pdb"
            # print(fpath)
            structure = esm.inverse_folding.util.load_structure(fpath, 'A')
            coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
            coords = torch.from_numpy(coords).cuda()
            rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
            # print(rep.shape)
            output = linear(rep.mean(0, keepdim=True))
            outputs.append(output)
            labels.append(torch.tensor(label).long().cuda())

            if len(outputs) == 4:
                outputs = torch.cat(outputs, 0)
                labels = torch.stack(labels, 0)
                # print(outputs.shape)
                # print(labels.shape)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                outputs = []
                labels = []
        if len(outputs) != 0:
            outputs = torch.cat(outputs, 0)
            labels = torch.stack(labels, 0)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs = []
            labels = []
        lr_scheduler.step()
        with torch.no_grad():
            for name, label in zip(split['test_names'], split['test_labels']):
                fpath = f"/home/xc4863/clean_datasets/d2/d2_clean/{name}/unrelaxed_model_1_ptm.pdb"
                structure = esm.inverse_folding.util.load_structure(fpath, 'A')
                coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
                rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)

                output = linear(rep.mean(0, keepdim=True))
                outputs.append(torch.argmax(output, 1))
                labels.append(torch.tensor(label).long().cuda())

        outputs = torch.cat(outputs, 0)
        labels = torch.stack(labels, 0)
        acc = (outputs == labels).float().sum() / labels.nelement()
        precision = ((outputs == labels).float() * (outputs == 1).float()).sum() / (outputs == 1).float().sum() 
        for i in range(5):
            poutputs = outputs[labels == i]
            print((poutputs == i).float().mean())
        print("ACC:", acc)
        print(precision)

        if acc > best_acc:
            best_acc = acc
        # torch.save({"model": model.state_dict(), "linear": linear.state_dict()}, f"dense_d2_esm1f_{split_num}.pth.tar")
    print(f"BEST_ACC: {best_acc}")