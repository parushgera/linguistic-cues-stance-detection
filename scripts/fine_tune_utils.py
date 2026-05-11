import optuna
import torch
import torch.optim as optim
import json
import os
import optuna.visualization as vis
import matplotlib.pyplot as plt

from utils import *

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset



# def compute_bert_embeddings(df, tokenizer, bert_model, max_seq_len=128):
#     """
#     Computes BERT embeddings for a given DataFrame and returns them in-memory.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing text data.
#         tokenizer (AutoTokenizer): Tokenizer for BERT model.
#         bert_model (AutoModel): BERT model for computing embeddings.
#         max_seq_len (int): Maximum token length for truncation & padding.

#     Returns:
#         torch.Tensor: Computed embeddings.
#         torch.Tensor: Corresponding labels (if available).
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     bert_model.to(device)
#     bert_model.eval()

#     embeddings = []
#     labels = [] if 'stance' in df else None  

#     for text in tqdm(df['text'], desc="Computing BERT Embeddings"):
#         encoded_input = tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=max_seq_len,  # ✅ Ensure max_length is an integer
#             return_tensors="pt"
#         ).to(device)

#         with torch.no_grad():
#             model_output = bert_model(**encoded_input)
#             text_embedding = model_output.last_hidden_state.mean(dim=1).squeeze(0)  # Mean pooling
        
#         embeddings.append(text_embedding.cpu())  
#         if labels is not None:
#             labels.append(df[df['text'] == text]['stance'].values[0])  

#     embeddings = torch.stack(embeddings)
#     labels = torch.tensor(labels) if labels else None
    
#     print(f"✅ Computed {len(embeddings)} embeddings (not saved)")
    
#     return embeddings, labels


def compute_bert_embeddings_token(df, tokenizer, bert_model, max_seq_len=128):
    """
    Computes BERT embeddings for RNN-based models like LSTMs and GRUs.
    
    Args:
        df (pd.DataFrame): DataFrame containing text data.
        tokenizer (AutoTokenizer): Tokenizer for BERT model.
        bert_model (AutoModel): BERT model for computing embeddings.
        max_seq_len (int): Maximum token length for truncation & padding.

    Returns:
        torch.Tensor: RNN-compatible embeddings (batch_size, sequence_length, embedding_dim).
        torch.Tensor: Corresponding labels (if available).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    embeddings = []
    labels = [] if 'stance' in df else None  

    for text in tqdm(df['text'], desc="Computing BERT Embeddings for RNN"):
        # Tokenization
        encoded_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            model_output = bert_model(**encoded_input)
            text_embedding = model_output.last_hidden_state.squeeze(0)  # Shape: (seq_len, embedding_dim)

        embeddings.append(text_embedding.cpu())  
        if labels is not None:
            labels.append(df[df['text'] == text]['stance'].values[0])  

    # Convert to Tensors
    embeddings = torch.stack(embeddings)  # Shape: (batch_size, seq_len, embedding_dim) -> (128, 128, 384)
    labels = torch.tensor(labels) if labels else None
    
    print(f"✅ Computed {len(embeddings)} RNN-compatible embeddings")

    return embeddings, labels


class PrecomputedBertDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        Dataset class that uses precomputed BERT embeddings stored in memory.
        
        Args:
            embeddings (torch.Tensor): Tensor of precomputed embeddings.
            labels (torch.Tensor or None): Corresponding labels (if available).
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        text_embedding = self.embeddings[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return text_embedding, label
        return text_embedding
    

def create_precomputed_loaders(embeddings, labels, batch_size, shuffle=True):
    """
    Creates DataLoader from precomputed embeddings (in-memory).
    
    Args:
        embeddings (torch.Tensor): Precomputed embeddings.
        labels (torch.Tensor or None): Corresponding labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader for training or evaluation.
    """
    dataset = PrecomputedBertDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader


 # **Return F1-score for optimization**




