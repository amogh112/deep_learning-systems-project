import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def how_sparse(model):
    pass

def prune(model, config):
    #currently focusing only on conv layers
    strategy = config['strategy']
    layers = config['layers']
    backbone = config['backbone']
    amount = config['amount']
    method = config['method']
    
    section_to_prune = None
    if backbone == True:
        #need model.module for DataParallel object
        section_to_prune = model.module.encoder
    else:
        section_to_prune = model.module.task_to_decoder
        #right now only considering the backbone
        #section_to_prune = model.task_to_decoder[task]
    
    if layers == []:
        #include all conv layers
        modules = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    else:
        modules = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d) and name in layers]
    
    if method == 'L1Unstructured':
        prune_fn = prune.l1_unstructured
    elif method == 'RandomUnstructured':
        prune_fn = prune.random_unstructured
    elif method == 'RandomStructured'

    if strategy == 'global':
        parameters_to_prune = []
        for module in modules:
            parameters_to_prune.append((module, 'weight'))
        parameters_to_prune = tuple(parameters_to_prune)
        print(parameters_to_prune)
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
    elif strategy == 'local':
        
        for module in modules:
            if method == 'L1Unstructured':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == 'RandomUnstructured':
                prune.random_unstructured(module, name='weight', amount=amount)
            elif module == 'Structured':
                #hardcoding the metric for comparison to be l2 norm
                #pruning along the first dimension - this is basically at a filter level
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    

    return model