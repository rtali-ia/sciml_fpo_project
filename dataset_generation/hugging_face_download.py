from huggingface_hub import login, snapshot_download

# Hugging Face access token (replace with your token)
hf_token = ""

# Login to Hugging Face using the token
print("Logging into Hugging Face...")
login(token=hf_token)

# Specify repository and folder details
repo_id = "BGLab/FlowBench"  # Repository ID on Hugging Face
dataset_path = "FPO_NS_2D_1024x256"  # Folder path within the repository
output_dir = "./downloaded_folder"  # Local directory to save the folder

# Download the entire repository or specific folder
print(f"Downloading folder '{dataset_path}' from repository '{repo_id}'...")
snapshot_download(repo_id, repo_type="dataset", local_dir=output_dir, allow_patterns=[f"{dataset_path}/*"])

print(f"Folder downloaded successfully to {output_dir}!")