import os
from os.path import join as pjoin
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer



class GigaHandsT2M(Dataset):
    """
    Core Dataset Loader for GigaHands T2M format.

    This class reads annotations from a JSONL file, finds the corresponding motion .npy files
    (either left or right hand), normalizes them using provided mean and std values, and returns
    them in the shape expected by the MDM model.

    Output:
        Each sample is a dictionary with:
            - 'inp': normalized motion tensor [263, 1, T]
            - 'text': corresponding textual annotation
            - 'lengths': number of frames
            - 'key': file path for reference

    This class is intended to be used internally by a wrapper that conforms to MDM's dataset expectations.
    """
    def __init__(self, root_dir, annotation_file, mean_std_dir, side='left', split='train', device='cpu',num_frames=120):
        assert side in ['left', 'right']
        self.side = side
        self.root_dir = root_dir
        self.device = device
        self.num_frames = num_frames
        self.fixed_len = num_frames
        self.max_text_len = 40
       


        self.mean = np.load(pjoin(mean_std_dir, f'mean_{side}.npy'))
        self.std = np.load(pjoin(mean_std_dir, f'std_{side}.npy'))

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        self.samples = []  # list of (motion_path, text)
        self._load_annotations(annotation_file, split)

    def _load_annotations(self, annotation_file, split):
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Loading GigaHands [{split}]"):
            ann = json.loads(line)
            scene = ann['scene']
            seq = ann['sequence']
            text_list = ann['rewritten_annotation']
            text = text_list[0]

            motion_path = pjoin(self.root_dir, scene, 'keypoints_3d', seq, f'dmvb_{self.side}.npy')
            if os.path.exists(motion_path):
                self.samples.append((motion_path, text))

        assert len(self.samples) > 0, "No valid samples found."

    def __len__(self):
        return len(self.samples)



    def __getitem__(self, idx):
        motion_path, text = self.samples[idx]
        motion = np.load(motion_path).astype(np.float32)

        # Normalize
        motion = (motion - self.mean) / (self.std + 1e-8)

        # Cut or pad motion to fixed length
        if self.fixed_len > 0:
            T = motion.shape[0]
            if T >= self.fixed_len:
                start = np.random.randint(0, T - self.fixed_len + 1)
                motion = motion[start:start + self.fixed_len]
            else:
                pad = np.zeros((self.fixed_len - T, motion.shape[1]), dtype=np.float32)
                motion = np.concatenate([motion, pad], axis=0)
            m_length = self.fixed_len
        else:
            m_length = motion.shape[0]

        # BERT-compatible dummy tokenization
        tokens = text.split()
        tokens = ['sos/OTHER'] + tokens[:self.max_text_len] + ['eos/OTHER']
        sent_len = len(tokens)
        tokens += ['unk/OTHER'] * (self.max_text_len + 2 - len(tokens))

        # Dummy embeddings and POS (you can skip glove)
        word_embeddings = np.random.randn(self.max_text_len + 2, 300).astype(np.float32)
        pos_one_hots = np.zeros((self.max_text_len + 2, 49), dtype=np.float32)

        return (
            torch.from_numpy(word_embeddings),          # [77, 300]
            torch.from_numpy(pos_one_hots),             # [77, 49]
            text,                                        # caption
            sent_len,                                    # caption length
            torch.from_numpy(motion),                   # [T, D]
            m_length,                                    # length
            '_'.join(tokens)                             # tokenized key
        )










class GigaHandsML3D(Dataset):
    def __init__(self, mode, datapath=None, split="train", **kwargs):
        self.mode = mode
        self.dataset_name = 'gigahands'
        self.dataname = 'gigahands'

        # Device
        device = kwargs.get('device', 'cpu')
        self.device = device

        # Absolute paths (edit if location changes)
        root_dir = r"D:\repos\mdm_custom_training\converted_motions\hand_poses"
        annotation_file = r"D:\repos\mdm_custom_training\converted_motions\annotations_v2.jsonl"
        mean_std_dir = r"D:\repos\mdm_custom_training\converted_motions\hand_poses\norm_stats"

        # Optional values
        fixed_len = kwargs.get('fixed_len', 0)
        use_cache = kwargs.get('use_cache', True)

        # Load mean/std
        side = 'left'  # or pass it as a parameter if needed

        mean = np.load(pjoin(mean_std_dir, f'Mean_{side}.npy'))
        std = np.load(pjoin(mean_std_dir, f'Std_{side}.npy'))

        self.mean = mean
        self.std = std

        # Fake opt to mimic original behavior
        self.opt = type('', (), {})()
        self.opt.fixed_len = fixed_len
        self.opt.max_motion_length = fixed_len if fixed_len > 0 else 196
        self.opt.unit_length = 4
        self.opt.max_text_len = 64
        self.opt.disable_offset_aug = False

        # Load GigaHandsT2M with left hand
        self.t2m_dataset = GigaHandsT2M(
            root_dir=root_dir,
            annotation_file=annotation_file,
            mean_std_dir=mean_std_dir,
            split=split,
            side='left',  # <-- make sure to use only left hand
            num_frames=fixed_len if fixed_len > 0 else 196,
            device=device
        )

        self.mean_gpu = torch.tensor(mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(std).to(device)[None, :, None, None]

        assert len(self.t2m_dataset) > 1, 'GigaHands dataset appears empty.'

    def __getitem__(self, idx):
        return self.t2m_dataset[idx]

    def __len__(self):
        return len(self.t2m_dataset)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--annotation_file', required=True)
    parser.add_argument('--mean_std_dir', required=True)
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    dataset = GigaHandsT2M(
        root_dir=args.root_dir,
        annotation_file=args.annotation_file,
        mean_std_dir=args.mean_std_dir,
        side=args.side,
        split=args.split
    )

    print(f"Loaded {len(dataset)} samples")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"[{i}] {sample['key']}: {sample['text']} → motion shape {sample['inp'].shape}")
