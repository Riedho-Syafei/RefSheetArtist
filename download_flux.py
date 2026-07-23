import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # forces regular download path

from huggingface_hub import snapshot_download

# Tell Hugging Face who you are
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE"

print("⏳ Starting download! This pulls about 20 gigabytes of files directly to your drive...")
snapshot_download(
    repo_id="black-forest-labs/FLUX.2-klein-4B",
    local_dir=r"C:\AI\models\FLUX.2-klein-4B",  # Where you want the files saved
    token=HF_TOKEN
)
print("✨ Complete! The model folder is completely ready at C:\\AI\\models\\FLUX.2-klein-4B")