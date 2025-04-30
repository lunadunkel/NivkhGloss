from huggingface_hub import hf_hub_download
import torch

def download_file_path(filename, token):
    downloaded_file_path = hf_hub_download(repo_id='lunadunkel/NivkhGloss', filename=filename, use_auth_token=token)
    rnn_checkpoint = torch.load(downloaded_file_path)
    return rnn_checkpoint