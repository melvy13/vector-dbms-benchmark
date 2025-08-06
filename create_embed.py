import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ------------------------------------
# 1. 'Unpickle' CIFAR-10 batch file
# ------------------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# ------------------------------------
# 2. Convert CIFAR-10 raw data to list of PIL images
# ------------------------------------
def load_images(data_dir):
    images = []
    labels = []
    for batch_id in range(1, 6): # data_batch_1 to data_batch_5
        batch_path = os.path.join(data_dir, f"data_batch_{batch_id}")
        batch = unpickle(batch_path)
        batch_data = batch[b'data']
        batch_labels = batch[b'labels']

        for i in range(len(batch_labels)):
            img_array = batch_data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # CHW -> HWC
            img = Image.fromarray(img_array)
            images.append(img)
            labels.append(batch_labels[i])

    return images, labels

# ------------------------------------
# 3. Generate embeddings with CLIP
# ------------------------------------
def generate_embeddings(images, model_name="clip-ViT-B-32"):
    model = SentenceTransformer(model_name)
    # Convert PIL images to embeddings
    embeddings = model.encode(images, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# ------------------------------------
# Main
# ------------------------------------
if __name__ == "__main__":
    image_dir = "cifar-10-batches-py" 
    images, labels = load_images(image_dir)

    print(f"Loaded {len(images)} images")

    embeddings = generate_embeddings(images)

    print(f"Embeddings shape: {embeddings.shape}")  # (50000, 512) for CLIP

    np.save("cifar10_embeddings.npy", embeddings)
    np.save("cifar10_labels.npy", np.array(labels))

    print("Saved embeddings to cifar10_embeddings.npy and labels to cifar10_labels.npy")

