"""
Download all 5 fine-tuned DeBERTa models from HuggingFace
Run this from your project root directory
"""

from huggingface_hub import snapshot_download
import os

REPO_ID = "zozbay/pii-detection-deberta-v3-large"

print(f"Downloading models from HuggingFace...")
print(f"Repository: {REPO_ID}\n")

try:
    # Download all models
    snapshot_download(
        repo_id=REPO_ID,
        local_dir="models",
        repo_type="model",
        resume_download=True  # Resume if interrupted
    )
    
    print("\n✓ Models downloaded successfully!")
    print("\nModel structure:")
    for i in range(5):
        model_path = f"models/fold_{i}"
        if os.path.exists(model_path):
            print(f"  ├── fold_{i}/ ✓")
        else:
            print(f"  ├── fold_{i}/ ✗ MISSING")
    
    print("\nYou can now run the evaluation pipeline!")
    
except Exception as e:
    print(f"\n✗ Error downloading models: {e}")
    print("\nAlternative: Download manually from:")
    print(f"https://huggingface.co/{REPO_ID}")