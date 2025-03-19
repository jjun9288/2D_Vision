import os
import numpy as np
import cv2
from PIL import Image
import json
from tqdm.notebook import tqdm
import faiss
import torch
import torchvision.transforms as T

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

transform_img = T.Compose([
    T.ToTensor(),
    T.Resize(224),
    T.CenterCrop(224),
    T.Normalize([0.5], [0.5])]
)

def load_img(img_path) -> torch.Tensor :
    input = Image.open(img_path)
    transformed_img = transform_img(input)[:3].unsqueeze(0)

    return transformed_img

def create_idx(files, model) -> faiss.IndexFlatL2 :
    idx = faiss.IndexFlatL2(384)
    all_embeddings = {}
    with torch.no_grad():
        for _, file in enumerate(tqdm(files)):
            embeddings = dino_v2_vits14(load_img(file).to(device))
            embedding = embeddings[0].cpu().numpy()
            all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()
            idx.add(np.array(embedding).reshape(1,-1))

    with open("all_embeddings.json", "w") as file:
        file.write(json.dumps(all_embeddings))
    
    faiss.write_index(idx, "bata.bin")

    return idx, all_embeddings

def search_idx(input_idx, input_embeddings, k=3) -> list:
    _, results = input_idx.search(np.array(input_embeddings[0].reshape(1,-1)), k)
    
    return results[0]

if __name__ == "__main__":

    # Load image
    cwd = os.getcwd()
    ROOT_DIR = os.path.join(cwd, "COCO-128-2/train/")
    files = os.listdir(ROOT_DIR)
    files = [os.path.join(ROOT_DIR, f) for f in files if f.lower().endswith(".jpg")]

    # Load model
    dino_v2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dino_v2_vits14.to(device)
    
    data_idx, all_embeddings = create_idx(files, dino_v2_vits14)

    input_file = "COCO-128-2/valid/000000000081_jpg.rf.5262c2db56ea4568d7d32def1bde3d06.jpg"
    input_img = cv2.resize(cv2.imread(input_file), (416,416))
    
    with torch.no_grad():
        embedding = dino_v2_vits14(load_img(input_file).to(device))
        results = search_idx(data_idx, np.array(embedding[0].cpu()).reshape(1,-1))

        for i, idx in enumerate(results):
            print(f"Image {i} : {files[idx]}")