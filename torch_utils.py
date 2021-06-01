
import numpy as np
import os
import random
import torch
import os


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def to_device(device, list_to_device):
    assert isinstance(list_to_device, list)

    for i, value_ in enumerate(list_to_device):
        if isinstance(value_, dict):
            for key in value_.keys():
                value_[key] = value_[key].to(device)
        elif isinstance(value_, torch.Tensor):
            list_to_device[i] = value_.to(device)
        else:
            continue
            raise NotImplementedError

    return list_to_device


def save_model(path, ep, model, optimizer):
    save_path = os.path.join(path, "saved_model.pt")
    torch.save({
        'ep': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print('models saved successfully. ({})'.format(path))


def load_model(path, model, optimizer):
    load_path = os.path.join(path, "saved_model.pt")
    if not os.path.exists(load_path):
        return False

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.eval()
    print('models loaded successfully. ({})'.format(path))
    return checkpoint['ep']


def use_profile():
    # # #### profiler start ####
    import builtins

    try:
        builtins.profile
    except AttributeError:
        # No line profiler, provide a pass-through version
        def profile(func):
            return func

        builtins.profile = profile


def calc_y(wgt0, y1, cost_r=0.):
    # wgt0: 0 ~ T-1 ,  y1 : 1 ~ T  => 0 ~ T (0번째 값은 0)
    y = dict()
    wgt1 = wgt0 * (1 + y1)
    # turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1]), axis=1), 0)
    turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1] / wgt1[:-1].sum(axis=1, keepdims=True)), axis=1), 0)
    y['before_cost'] = np.insert(np.sum(wgt1, axis=1) - 1, 0, 0)
    y['after_cost'] = np.insert(np.sum(wgt1, axis=1) - 1 - turnover * cost_r, 0, 0)

    return y, turnover


def collect_result_file_paths(dir_path):

    result_files = []
    file_names = os.listdir(dir_path)
    for name in file_names:
        file_path = os.path.join(dir_path, name)
        if os.path.isdir(file_path):
            result_files += collect_result_file_paths(file_path)

        if '0000' in name:
            result_files.append(os.path.join(dir_path, name))

    return result_files


def collect_result_files(root_path, out_path):
    """
    collect_result_files('D:/projects/asset_allocation/out/newmodel_eq', './out/abc')
    """
    from shutil import copyfile

    file_paths = collect_result_file_paths(root_path)
    os.makedirs(out_path, exist_ok=True)

    for file_path in file_paths:
        new_path = file_path.replace(root_path, out_path)
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        copyfile(file_path, new_path)
