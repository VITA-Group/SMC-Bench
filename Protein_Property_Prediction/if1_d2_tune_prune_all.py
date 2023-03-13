import numpy as np
import esm
import pickle
import torch
import torch.nn as nn
import sys
from esm.modules import TransformerLayer
from torch.nn.utils import prune
from masking import CosineDecay, Masking
split_num = sys.argv[1]
split = pickle.load(open(f"/home/xc4863/clean_datasets/d2/d2_{split_num}_classification.pkl", "rb"))
backbone_lr = float(sys.argv[3])
lr = float(sys.argv[2])
epoch = int(sys.argv[4])
pruning_ratio=float(sys.argv[5])
pruning_method=str(sys.argv[6])
init_method=str(sys.argv[7])
sparse_mode=str(sys.argv[8])


model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.cuda()
linear = nn.Sequential( nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 5)).cuda() 
optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=5e-2)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=int(epoch))
backbone_optimizer = torch.optim.AdamW(model.parameters(), lr=backbone_lr, weight_decay=5e-2)
best_acc = 0

decay = CosineDecay(0.5, len(split['train_names']) // 4 * epoch)
mask = Masking(optimizer, prune_rate_decay=decay, prune_rate=0.5,
                       sparsity=pruning_ratio, prune_mode=pruning_method,
                       growth_mode='gradient', redistribution_mode='none', sparse_init=init_method,
                       sparse_mode=sparse_mode, update_frequency=500)
mask.add_module(model)

def snip(keep_ratio, masks):
    outputs = []
    labels = []
    for batch_idx, (name, label) in enumerate(zip(split['train_names'], split['train_labels'])):
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
            outputs = []
            labels = []
        if batch_idx > 10: break
    grads_abs = []
    for name, m in model.named_modules():
        if name + ".weight" in masks:
            grads_abs.append(torch.clone(m.weight.grad).detach().abs_())

    # normalize score
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep+1, sorted=True)
    acceptable_score = threshold[-1]
    layer_wise_sparsities = []
    for g in grads_abs:
        mask_ = (g > acceptable_score).float()
        layer_wise_sparsities.append(mask_)

    model.zero_grad()

    return layer_wise_sparsities

if mask.sparse_init == 'snip':
    mask.init_growth_prune_and_redist()
    layer_wise_sparsities = snip(1 - mask.sparsity, mask.masks)
    for snip_mask, name in zip(layer_wise_sparsities, mask.masks):
        mask.masks[name][:] = snip_mask
    mask.apply_mask()
    mask.print_status()
else:
    mask.init(model=model, train_loader=None, device=mask.device,
                      mode=mask.sparse_init, density=(1 - mask.sparsity))

for epoch in range(epoch):
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
            
            mask.step()
            optimizer.zero_grad()
            outputs = []
            labels = []
    if len(outputs) != 0:
        outputs = torch.cat(outputs, 0)
        labels = torch.stack(labels, 0)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        mask.step()
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