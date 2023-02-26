from __future__ import print_function
import torch
import math
import copy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from funcs import redistribution_funcs, growth_funcs, prune_funcs



class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1, init_step=0):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)
        self.T_max = T_max
        if init_step!=0:
            for i in range(init_step):
                self.cosine_stepper.step()
    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']


class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)
    """
    def __init__(self, optimizer, prune_rate_decay, prune_rate=0.5, sparsity=0.0, prune_mode='magnitude',
                 growth_mode='random', redistribution_mode='momentum', verbose=False, fp16=False,
                 args=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.fix = args.fix
        self.noembed = args.noembed
        self.sparse_init = args.sparse_init
        self.sparse_mode = args.sparse_mode
        self.update_frequency = args.update_frequency
        self.sparsity = sparsity
        self.device = torch.device('cuda')
        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose
        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        # parameters for GMP
        self.total_step = self.prune_rate_decay.T_max
        self.final_prune_time = int(self.total_step * args.final_prune_time)
        self.initial_prune_time = int(self.total_step * args.initial_prune_time)

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.names = []
        self.optimizer = optimizer
        self.baseline_nonzero = None

        # stats
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}
        self.prune_rate = prune_rate
        self.steps = 0
        self.half = fp16
        self.name_to_32bit = {}

        if self.fix:
            self.update_frequency = None


    def add_module(self, modules):
        self.modules = modules
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if len(tensor.size()) == 2 or len(tensor.size()) == 4:
                    self.names.append(name)
                    self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        if self.noembed:
            print('Removing embeddings layers')
            self.remove_weight_partial_name('embeddings')

        if self.sparse_init == 'snip' or (not self.fix):
            print('Removing bert_model.pooler')
            self.remove_weight_partial_name('pooler')
            print('Removing merge layers')
            self.remove_weight_partial_name('merge')


    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)
                    # self.remove_weight_partial_name(name, verbose=self.verbose)

    def print_status(self):
        total_size = 0
        sparse_size = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                dense_weight_num = weight.numel()
                sparse_weight_num = (weight != 0).sum().int().item()
                total_size += dense_weight_num
                sparse_size += sparse_weight_num
                layer_density = sparse_weight_num / dense_weight_num
                print(f'sparsity of layer {name} with tensor {weight.size()} is {1-layer_density}')
        print('Final sparsity level of {0}: {1}'.format(self.sparsity, 1 - sparse_size / total_size))

    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True

    def init(self, model, train_loader , device, mode='snip', density=0.05, erk_power_scale=1.0):
        self.init_growth_prune_and_redist()

        if mode == 'dense':
            print('initialized with dense model')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).to(device)

        if mode == 'one_shot_gm':
            print('initialize by one_shot_gm')
            self.baseline_nonzero = 0
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) > acceptable_score).float().data.to(device)

        elif mode == 'random':
            print('initialize by random pruning')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (torch.rand(weight.shape) < density).float().data.to(device)


        if mode == 'iterative_gm':
            print('initialized by iterative_gm')
            total_num_nonzoros = 0
            dense_nonzeros = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = (weight != 0).cuda()
                    self.name2nonzeros[name] = (weight != 0).sum().item()
                    total_num_nonzoros += self.name2nonzeros[name]
                    dense_nonzeros += weight.numel()
                    print(f'sparsity of layer {name} is {self.name2nonzeros[name]/weight.numel()}')

            print(f'sparsity level of current model is {1-total_num_nonzoros/dense_nonzeros}')

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(total_num_nonzoros * density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) > acceptable_score).float().data.to(device)

        if mode == 'uniform':
            print('initialized with uniform')
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.to(device)
                    self.baseline_nonzero += weight.numel()*density

        elif mode == 'resume':
            print('initialized with resume')
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    self.masks[name][:] = (weight != 0.0).float().data.to(device)
                    self.baseline_nonzero += weight.numel()*density


        elif mode == 'ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            self.baseline_nonzero = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
                self.baseline_nonzero += weight.numel() * density
            is_epsilon_valid = False

            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros

                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale

                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(self.device)

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")



    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')


    def step(self):
        # self.optimizer.step()
        self.apply_mask()

        # decay the adaptation rate for better results
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        self.steps += 1

        if self.update_frequency is not None:
            if self.sparse_mode == 'GMP':
                if self.steps >= self.initial_prune_time and self.steps < self.final_prune_time and self.steps % self.update_frequency == 0:
                    print('*********************************Gradual Magnitude Pruning***********************')
                    current_prune_rate = self.gradual_pruning_rate(self.steps, 0.0, self.sparsity,
                                                                   self.initial_prune_time, self.final_prune_time)
                    self.gradual_magnitude_pruning(current_prune_rate)
                    self.print_status()
            else:
                if self.steps % self.update_frequency == 0:
                    print('*********************************Dynamic Sparsity********************************')
                    self.truncate_weights()
                    self.print_nonzero_counts()


    def apply_mask(self):

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                        # if 'momentum_buffer' in self.optimizer.state[tensor]:
                        #     self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]

    def truncate_weights(self):

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                # prune
                new_mask = self.prune_func(self, mask, weight, name)
                removed = self.name2nonzeros[name] - new_mask.sum().item()
                self.name2removed[name] = removed
                self.masks[name][:] = new_mask

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()
                # growth
                new_mask = self.growth_func(self, name, new_mask, math.floor(self.name2removed[name]), weight)
                self.masks[name][:] = new_mask.float()

        self.apply_mask()

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                               num_nonzeros / float(mask.numel()))
                print(val)

        print('Prune rate: {0}\n'.format(self.prune_rate))

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                # print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    def synchronism_masks(self):

        for name in self.masks.keys():
            torch.distributed.broadcast(self.masks[name], src=0, async_op=False)

    def gradual_pruning_rate(self,
            step: int,
            initial_threshold: float,
            final_threshold: float,
            initial_time: int,
            final_time: int,
    ):
        if step <= initial_time:
            threshold = initial_threshold
        elif step > final_time:
            threshold = final_threshold
        else:
            mul_coeff = 1 - (step - initial_time) / (final_time - initial_time)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)

        return threshold

    def gradual_magnitude_pruning(self, current_pruning_rate):
        weight_abs = []
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                weight_abs.append(torch.abs(weight))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        num_params_to_keep = int(len(all_scores) * (1 - current_pruning_rate))

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name] = ((torch.abs(weight)) > acceptable_score).float().data.to(self.device)
        self.apply_mask()

