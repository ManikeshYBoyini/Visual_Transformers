# 🧠 Vision Transformer (ViT) - Python Demo

This project is a hands-on demonstration of the **Vision Transformer (ViT)** architecture for image classification. The Vision Transformer, introduced in the groundbreaking paper [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929), was the first Transformer-based model to achieve competitive results on the ImageNet dataset.

This notebook-based demo walks you through a step-by-step implementation of the ViT model — from patch extraction to classification using a Transformer encoder — and ends with leveraging a pre-trained ViT model from Google trained on ImageNet-21k.

---

## 📚 Overview

Vision Transformer (ViT) treats an image as a sequence of patches, much like how words are treated in NLP. Instead of using convolutions, it applies a Transformer architecture directly to image patches.

### How ViT Works

1. **Patch Splitting**: The input image (`224x224x3`) is divided into fixed-size patches (`16x16`), producing `14x14 = 196` patches.
2. **Patch Embedding**: Each patch is flattened and passed through a linear projection to obtain a `768`-dimensional embedding vector.
3. **Position Embedding & [CLS] Token**: 
   - A learnable `[class]` token is prepended to the sequence.
   - Positional embeddings are added to preserve spatial information.
4. **Transformer Encoder**: The patch + position embeddings are passed through a standard Transformer encoder.
5. **Classification**: The output corresponding to the `[class]` token is passed through an MLP head to produce the final class prediction.

---

## 📂 Project Structure

This notebook is divided into the following parts:

1. **Setup** – Installing and importing necessary libraries.
2. **Patch Extraction** – Dividing the image into patches.
3. **Patch Encoding** – Creating patch embeddings and adding positional encoding.
4. **Multilayer Perceptron (MLP)** – Simple MLP used in the classification head.
5. **Transformer Encoder** – The core Transformer model.
6. **Putting it All Together** – Combining all the pieces into the full ViT model.
7. **Using Google's Pretrained ViT** – Leveraging pretrained weights trained on ImageNet-21k for inference.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow or PyTorch (depending on your implementation)
- Jupyter Notebook
- NumPy, Matplotlib

You can install the dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
````

Or, if using PyTorch and HuggingFace Transformers:

```bash
pip install torch torchvision transformers
```

### Running the Notebook

1. Clone this repository:

```bash
git clone https://github.com/yourusername/vit-python-demo.git
cd vit-python-demo
```

2. Launch the notebook:

```bash
jupyter notebook ViT_Demo.ipynb
```

---

## 🧪 Using Pretrained ViT

Training ViT from scratch requires large datasets and compute resources. Instead, we demonstrate the use of **Google's pretrained ViT model**, trained on **ImageNet-21k** (14M images, 21k classes).

You can load and use it via HuggingFace Transformers:

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
```

---

## 📈 Reference

* [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) – Original ViT paper.
* [ImageNet-21k Dataset](https://www.image-net.org/)

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome! Feel free to fork the repo and open a pull request.

---


## 🙌 Acknowledgements

* Google Research for pioneering ViT
* HuggingFace Transformers for providing pretrained models

