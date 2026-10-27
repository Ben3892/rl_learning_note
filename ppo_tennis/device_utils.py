"""Explicit CPU/CUDA selection: never silently fall back from requested CUDA."""

import torch


def resolve_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError('device 必须为 cpu、cuda、cuda:N 或 auto') from error
    if device.type == 'cpu' and device.index is None:
        return 'cpu'
    if device.type != 'cuda':
        raise ValueError('本项目支持 cpu、cuda、cuda:N 或 auto')
    if not torch.cuda.is_available():
        raise ValueError('指定了 CUDA，但 PyTorch 无法使用 CUDA。请在 NVIDIA 机器上安装 '
                         'CUDA 版 PyTorch，并检查驱动及 CUDA_VISIBLE_DEVICES；'
                         '本机调试请显式使用 --device cpu。')
    index = torch.cuda.current_device() if device.index is None else device.index
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f'cuda:{index} 不存在，当前进程只看到 {count} 张 GPU；'
                         '编号以 CUDA_VISIBLE_DEVICES 映射后的设备为准。')
    return f'cuda:{index}'


def device_info(requested, actual):
    device = torch.device(actual)
    return dict(requested=requested, actual=str(device), torch_version=torch.__version__,
                torch_cuda_version=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
