import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join as pjoin

class GigaHandsMDMWrapper(Dataset):
    def __init__(self, mode, root_dir, annotation_file, mean_std_dir, split="train", num_frames=60, device="cpu", **kwargs):
        self.mode = mode
        self.device = device
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.mean_std_dir = mean_std_dir
        self.num_frames = num_frames
        self.split = split

        # Load mean/std
        self.mean = np.load(pjoin(mean_std_dir, 'Mean.npy'))
        self.std = np.load(pjoin(mean_std_dir, 'Std.npy'))

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        # Load annotations
        with open(annotation_file, 'r') as f:
            lines = [json.loads(l) for l in f]
        self.samples = [
            (pjoin(root_dir, item["scene"], "keypoints_3d", item["sequence"], "dmvb_left.npy"), item["description"])
            for item in lines if item.get("split", "train") == split
        ]

        assert len(self.samples) > 0, f"No data found for split {split}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        motion_path, text = self.samples[idx]
        motion = np.load(motion_path).astype(np.float32)  # [T, D]

        motion = (motion - self.mean) / (self.std + 1e-8)

        # Crop or pad to num_frames
        T = motion.shape[0]
        if T >= self.num_frames:
            motion = motion[:self.num_frames]
        else:
            pad = np.zeros((self.num_frames - T, motion.shape[1]), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        motion = torch.tensor(motion.T).unsqueeze(1)  # [D, 1, T]

        return {
            'inp': motion,
            'cond': {
                'y': {
                    'text': text,
                    'length': len(text.split())
                }
            },
            'key': motion_path
        }
