# datasets/gigahands_interface.py
import os
import yaml
from torch.utils.data import DataLoader
from dataset_api.dataset_interface import DatasetInterface, DatasetFunctions
from data_loaders.humanml.data.Gigadataset_loader import GigaHandsML3D

class GigaHandsInterface(DatasetInterface):
    def __init__(self):
        cur_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(cur_dir, "gigahands.yaml")
        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)
    
        

    def get_function(self, name: str):
        fn = DatasetFunctions(name)

        if fn == DatasetFunctions.GET_LOADER:
            def get_loader(args):
                dataset = GigaHandsML3D(
                    mode='train',
                    split='train',
                    device=args.device,
                    fixed_len=args.pred_len + args.context_len
                )
                return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
            return get_loader

        elif fn == DatasetFunctions.EVALUATE:
            from eval import eval_gigahands
            return eval_gigahands.evaluate

        elif fn == DatasetFunctions.COLLATE:
            def default_collate(batch):
                return tuple(zip(*batch))
            return default_collate

        elif fn == DatasetFunctions.WRAPPER:
            return lambda device: None  # GigaHands doesn’t use evaluator wrapper

        elif fn == DatasetFunctions.CONFIG:
            return self.config

        raise NotImplementedError(f"GigaHandsInterface: function {name} not implemented")



    def get_config(self) -> dict:
        return self.config
