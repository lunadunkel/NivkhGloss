from huggingface_hub import HfApi

def upload_checkpoint(path, token, name='best_checkpoint.pth'):
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=name,
        repo_id="lunadunkel/NivkhGloss",
        repo_type="model"
    )