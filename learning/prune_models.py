import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def how_sparse(modules):
    total_sparsity = 0
    total_params = 0
    for module in modules:
        if isinstance(module, torch.nn.Conv2d):
            module_sparsity = float(torch.sum(module.weight == 0))
            total_module_params = float(module.weight.nelement())
            total_sparsity += module_sparsity
            total_params += total_module_params
            
        print('Sparsity in {} = {}'.format(module, float(module_sparsity/total_module_params)))
    print('Total sparsity in selected modules is {}'.format(float(total_sparsity/total_params)))
    
def how_sparse_filters(module):
    if isinstance(module, torch.nn.Conv2d):
        
        num_filters = module.weight.shape[0]
        empty_filters = 0
        for idx in range(num_filters):
            filter = module.weight[idx]
            if torch.sum(module.weight[idx] != 0) == 0:
                empty_filters += 1
            
        print('Fraction of empty filters is {}'.format(float(empty_filters/module.weight.shape[0])))

def theoritical_flops(module):
    #TO DO
    pass


def prune_model(model, config):
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
    

    if strategy == 'global':
        parameters_to_prune = []
        for module in modules:
            parameters_to_prune.append((module, 'weight'))
        parameters_to_prune = tuple(parameters_to_prune)

        if method == 'L1Unstructured':
            pruning_method = prune.L1Unstructured
        elif method == 'RandomUnstructured':
            pruning_method = prune.RandomUnstructured
            
        #pruning method used should be unstructured
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=amount
        )
        
        
    elif strategy == 'local':
        for module in modules:
            if method == 'L1Unstructured':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == 'RandomUnstructured':
                prune.random_unstructured(module, name='weight', amount=amount)

            elif method == 'Structured':
                
                #hardcoding the metric for comparison to be l2 norm
                #pruning along the first dimension - this is basically at a filter level
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                how_sparse_filters(module)
                
    how_sparse(modules)

    return model