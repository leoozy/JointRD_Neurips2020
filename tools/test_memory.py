import torch, pdb
from prettytable import PrettyTable
table = PrettyTable(["module_architecture", "input_shape", "output_shape", "mem_pre", "mem_after", "max_mem"])
res = {}
    
def hook_fn_pre_forward(module, input):

    res["module_architecture"] = module.extra_repr()
    res["input_shape"] = str(input[0].shape)[10:]
    res["mem_pre_allo"] = torch.cuda.memory_allocated()
    torch.cuda.reset_max_memory_allocated()
    
def hook_fn_forward(module, input, output):
    
    res["output_shape"] = str(output[0].shape)[10:]
    res["mem_after_allo"] = torch.cuda.memory_allocated()
    res["max_mem"] = torch.cuda.max_memory_allocated()
    table.add_row([res["module_architecture"],
                   res["input_shape"],
                   res["output_shape"],
                   res["mem_pre_allo"],
                   res["mem_after_allo"],
                   res["max_mem"]])
    
    
def bound_pre_forward(model):

    modules = model.named_modules()  
    for name, module in modules:
        module.register_forward_pre_hook(hook_fn_pre_forward)
        module.register_forward_hook(hook_fn_forward)
        
def print_table():
    print(table)