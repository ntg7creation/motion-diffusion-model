from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate

import platform

NUM_WORKERS = 0 if platform.system() == 'Windows' else 8  # or any number you find optimal on Linux


def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "gigahands":
        from data_loaders.humanml.data.Gigadataset_loader import GigaHandsML3D
        return GigaHandsML3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')
    
def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit", "gigahands"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, cache_path=None): 
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive)
    elif name == "gigahands":
        dataset = DATA(
            mode=hml_mode,
            annotation_file=r"D:\repos\mdm_custom_training\converted_motions\annotations_v2.jsonl",
            root_dir=r"D:\repos\mdm_custom_training\converted_motions\hand_poses",
            mean_std_dir=r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats",
            split=split,
            num_frames=num_frames,
            device=device
        )

    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS,
          drop_last=True, collate_fn=collate
    )

    return loader