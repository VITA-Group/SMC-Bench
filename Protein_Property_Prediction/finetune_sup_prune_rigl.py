#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer
from masking import Masking, CosineDecay

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )

    parser.add_argument(
        "--split_file",
        type=str,
        help="fold",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        help="num_classes",
        default=2, 
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rates",
        default=2e-2, 
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        help="learning rates",
        default=1e-6, 
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--idx", type=str, default='0')
    parser.add_argument("--pruning_ratio", type=float, default=0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pruning_method", type=str, default='magnitude', choices=['magnitude', 'random', 'snip'])
    parser.add_argument("--init_method", type=str, default='one_shot_gm', choices=['one_shot_gm', 'random'])
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--label-offset", type=int, default=0)
    
    return parser

def pruning_model(model, px, method='omp'):
    
    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
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

def set_seed(args):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main(args):

    set_seed(args)
    best = 0
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location, num_classes=args.num_classes)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    import sys

    train_set = PickleBatchedDataset.from_file(args.split_file, True, args.fasta_file)
    test_set = PickleBatchedDataset.from_file(args.split_file, False, args.fasta_file)
    #train_batches = train_set.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, collate_fn=alphabet.get_batch_converter(), batch_size=args.batch_size, shuffle=True#batch_sampler=train_batches
    )
    #print(f"Read {args.fasta_file} with {len(train_sets[0])} sequences")

    #test_batches = test_set.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    test_data_loader = torch.utils.data.DataLoader(
        test_set, collate_fn=alphabet.get_batch_converter(), #batch_sampler=test_batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.backbone_lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.backbone_lr, steps_per_epoch=1, epochs=int(3))
    linear = nn.Sequential( nn.Linear(1280, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, args.num_classes)).cuda()
    head_optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr)
    head_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(head_optimizer, max_lr=args.lr, steps_per_epoch=1, epochs=int(3))
    steps = 0
    if args.checkpoint:
        checkpoints = torch.load(args.checkpoint)
        model.load_state_dict(checkpoints['model'])
        linear.load_state_dict(checkpoints['linear'])
    # pruning_model(model, args.pruning_ratio, args.pruning_method)

    # optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
    decay = CosineDecay(0.5, len(train_data_loader) * 4)
    mask = Masking(optimizer, prune_rate_decay=decay, prune_rate=0.5,
                           sparsity=args.pruning_ratio, prune_mode=args.pruning_method,
                           growth_mode='gradient', redistribution_mode='none', sparse_init=args.init_method)
    mask.add_module(model)
    mask.init(model=model, train_loader=None, device=mask.device,
                          mode=mask.sparse_init, density=(1 - mask.sparsity))
    for epoch in range(3):
        model.train()
        for batch_idx, (labels, strs, toks) in enumerate(train_data_loader):
            steps += 1
            with torch.autograd.set_detect_anomaly(True):
                print(
                    f"Processing {batch_idx + 1} of {len(train_data_loader)} batches ({toks.size(0)} sequences)"
                )
                toks = toks.cuda()
                
                if args.truncate:
                    toks = toks[:, :1022]
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)

                hidden = out['hidden']
                logits = linear(hidden)
                labels = torch.tensor(labels).cuda().long() - args.label_offset
                loss = (torch.nn.functional.cross_entropy(logits.reshape(-1, args.num_classes), labels.reshape(-1)))
                        
                loss.backward()
                optimizer.step()
                mask.step()
                
                model.zero_grad()
                print(loss.item())
            if steps % args.eval_freq == 0:
                mask.print_status()
                model.eval()
                with torch.no_grad():
                    outputs = []
                    tars = []
                    for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
                        print(
                            f"Processing {batch_idx + 1} of {len(test_data_loader)} batches ({toks.size(0)} sequences)"
                        )
                        if torch.cuda.is_available() and not args.nogpu:
                            toks = toks.to(device="cuda", non_blocking=True)
                        # The model is trained on truncated sequences and passing longer ones in at
                        # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
                        if args.truncate:
                            toks = toks[:, :1022]
                        out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)
                        hidden = out['hidden']
                        logits = linear(hidden)
                        labels = torch.tensor(labels).cuda().long() - args.label_offset
                        outputs.append(torch.argmax(logits.reshape(-1, args.num_classes), 1).view(-1))
                        tars.append(labels.reshape(-1))
                    import numpy as np
                    outputs = torch.cat(outputs, 0)
                    tars = torch.cat(tars, 0)
                    print("EVALUATION:", float((outputs == tars).float().sum() / tars.nelement()))
                    acc = (outputs == tars).float().sum() / tars.nelement()
                    if acc > best:
                        best = acc
        lr_scheduler.step()
        head_lr_scheduler.step()
    print(best)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
