import torch
import json
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from dataset_api.dataset_registry import DatasetInterfaceRegistry

# --- SETTINGS ---
DATASET = "gigahands"
CKPT_PATH = r".save/test_run/model000001000.pt"   # update path if needed
TEXT_PROMPT = "move the hand"
NFRAMES = 60   # should match training pred_len/context_len
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1) Load dataset interface & config
    dataset_interface = DatasetInterfaceRegistry.get(DATASET)
    args = dataset_interface.get_config()

    # Merge minimal runtime args
    args["dataset"] = DATASET
    args["device"] = DEVICE

    # 2) Create model + diffusion
    model, diffusion = create_model_and_diffusion(
        type("Args", (), args), data=None
    )
    model.to(DEVICE)

    # 3) Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    load_model_wo_clip(model, ckpt)
    model.eval()

    # 4) Dummy diffusion timestep (e.g., start from step 0)
    timesteps = torch.zeros(1, dtype=torch.long, device=DEVICE)

    # 5) Prepare dummy input motion (zeros, shape [bs, njoints, nfeats, nframes])
    njoints, nfeats = model.njoints, model.nfeats
    x = torch.zeros(1, njoints, nfeats, NFRAMES, device=DEVICE)

    # 6) Conditioning dict with text
    cond = {
        "text": [TEXT_PROMPT],
        "mask": torch.ones(1, 1, 1, NFRAMES, device=DEVICE, dtype=torch.bool)
    }

    # 7) Forward pass
    with torch.no_grad():
        out = model(x, timesteps, cond)

    print("=== MODEL OUTPUT ===")
    print("Shape:", out.shape)  # [1, njoints, nfeats, nframes]
    print("Sample values:", out[0, :, :, :5])  # first 5 frames

if __name__ == "__main__":
    main()
