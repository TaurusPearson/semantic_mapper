# Run this inside your python environment
from huggingface_hub import hf_hub_download
import shutil
import os

os.makedirs("weights", exist_ok=True)

print("Downloading SAM3 weights (sam3.pt)...")
# The correct file is sam3.pt, NOT safetensors
# Requires access approval at: https://huggingface.co/facebook/sam3
try:
    downloaded_path = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
    destination = "weights/sam3.pt"
    shutil.copy(downloaded_path, destination)
    print(f"Success! File saved to {destination}")
except Exception as e:
    print(f"Error: {e}")
    print("\nIf 401/403 error: You need to request access at https://huggingface.co/facebook/sam3")
    print("Then run: huggingface-cli login")