import os
import sys
import json
import re
import string
from pathlib import Path

# Make the `baselines` package (at the repo root) importable regardless of CWD.
_THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(_THIS_FILE.parent.parent))
sys.path.insert(0, str(_THIS_FILE.parent))

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import baselines.all_baselines as md
from mappings import targets, semeval_labels, wtwt_labels
from collections import defaultdict, Counter
from sklearn.metrics import classification_report
import gc
import copy
import random
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import nn
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from fine_tune_utils import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Repo-root-relative paths so the script can be invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"

with open(CONFIG_DIR / "best_parameters.json", "r") as f:
    best_params_all = json.load(f)
    
    
# best_params_all = {
#   "BiGRU": {
#     "whole": {
#       "epochs": 5,
#       "batch_size": 16,
#       "lr": 0.0001,
#       "dropout": 0.4,
#       "hidden_size": 128,
#       "num_layers": 1
#     }
#   },
#   "ATBiGRU": {
#     "whole": {
#       "epochs": 5,
#       "batch_size": 16,
#       "lr": 0.0001,
#       "dropout": 0.4,
#       "hidden_size": 128,
#       "num_layers": 1
#     }
#   },
#   "BiLSTM": {
#     "whole": {
#       "epochs": 5,
#       "batch_size": 16,
#       "lr": 0.0001,
#       "dropout": 0.4,
#       "hidden_size": 128,
#       "num_layers": 1
#     }
#   },
#   "ATBiLSTM": {
#     "whole": {
#       "epochs": 5,
#       "batch_size": 16,
#       "lr": 0.0001,
#       "dropout": 0.4,
#       "hidden_size": 128,
#       "num_layers": 1
#     }
#   },
#   "KimCNN": {
#     "whole": {
#       "epochs": 5,
#       "batch_size": 16,
#       "lr": 0.0001,
#       "dropout": 0.5,
#       "fc1_size": 100
#     }
#   },
#   "BERTModel": {
#     "whole": {
#       "epochs": 5,
#       "batch_size": 16,
#       "lr": 0.00002,
#       "dropout": 0.2
#     }
#   }
# }


def get_dataloader(model_class, batch_size):
    # This function now ONLY returns train and validation loaders.
    if model_class in bert_models:
        return (
            create_transformer_loaders(df_train, batch_size, bert_tokenizer, shuffle=True),
            create_transformer_loaders(df_val, batch_size, bert_tokenizer, shuffle=False),
        )
    else:
        return (
            create_precomputed_loaders(train_embeddings, train_labels, batch_size=batch_size, shuffle=True),
            create_precomputed_loaders(val_embeddings, val_labels, batch_size=batch_size, shuffle=False),
        )
    
def initialize_model(model_class):
    if model_class in cnn_models:
        return model_class(num_classes, dropout, fc1_size).to(device)
    elif model_class in bert_models:
        return model_class(base_model, num_classes).to(device)
    else:
        return model_class(num_classes, hidden_size, dropout, num_layers).to(device)
    

interested_labels = [0,1,2]
label_names = ['AGAINST', 'FAVOR', 'NONE']
df_master = pd.read_csv(DATASET_DIR / 'all_combined.csv')
semeval_labels_list = ['AGAINST', 'FAVOR', 'NONE']
semeval_labels = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

wtwt_map = {
    'support': 'FAVOR',
    'refute': 'AGAINST',
    'comment': 'NONE',
    'unrelated': 'NONE'
}
df_master['stance'] = df_master['stance'].replace(wtwt_map)
df_source_train = filter_df_on_target_dataset_split(target=None, dataset=None, df=df_master, split='train')
df_source_test = filter_df_on_target_dataset_split(target=None, dataset=None, df=df_master, split='test')
print(f"✅ Training examples: {len(df_source_train)}")
print(f"✅ Test examples: {len(df_source_test)}")
print("🔄 Mapping stance labels...")
df_source_train = map_stance_labels(df_source_train)
df_train, df_val, _ = stratified_train_val_test_split(df_source_train, df_source_test)
df_except_wtwt_test = pd.read_csv(PROCESSED_DIR / "except_wtwt_test.csv")
df_except_wtwt_test = map_stance_labels(df_except_wtwt_test)
df_wtwt_test_processed = pd.read_csv(PROCESSED_DIR / "wtwt_test_processed.csv")
print("\n📊 Label Distribution in Train, Validation, and Test Sets:")
print("----------------------------------------------------")
print(f"🔹 Train Distribution:\n{df_train['stance'].value_counts(normalize=True) * 100}\n")
print(f"🔹 Validation Distribution:\n{df_val['stance'].value_counts(normalize=True) * 100}\n")
print(f"🔹 Test Distribution:\n{df_except_wtwt_test['stance'].value_counts(normalize=True) * 100}\n")
print(f"🔹 Test Distribution:\n{df_wtwt_test_processed['stance'].value_counts(normalize=True) * 100}\n")


# Create embeddings for both test sets
SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
sentence_transformer_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER)
sentence_transformer_bert_model = AutoModel.from_pretrained(SENTENCE_TRANSFORMER)
train_embeddings, train_labels = compute_bert_embeddings_token(df_train, sentence_transformer_tokenizer, sentence_transformer_bert_model)
val_embeddings, val_labels = compute_bert_embeddings_token(df_val, sentence_transformer_tokenizer, sentence_transformer_bert_model)
wtwt_test_embeddings, wtwt_test_labels = compute_bert_embeddings_token(df_wtwt_test_processed, sentence_transformer_tokenizer, sentence_transformer_bert_model)
except_wtwt_test_embeddings, except_wtwt_test_labels = compute_bert_embeddings_token(df_except_wtwt_test, sentence_transformer_tokenizer, sentence_transformer_bert_model)

# --- SETUP (No Changes Here) ---
num_classes = df_train['stance'].nunique()
print("Here is num_classes", num_classes)
stance_counts = df_train['stance'].value_counts().sort_index()

class_weights = 1.0 / stance_counts
class_weights = class_weights / class_weights.sum()
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)

if num_classes == 2:
    loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor[1]).to(device)
else:
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

# --- MODIFIED: Initialize containers for results and predictions for both test sets ---
results = []
all_wtwt_predictions = {}
all_except_wtwt_predictions = {}

rnn_models = [md.BiGRU, md.ATBiGRU, md.BiLSTM, md.ATBiLSTM]
cnn_models = [md.KimCNN]
bert_models = [md.BERTModel]
models = rnn_models + cnn_models + bert_models
BERT_MODEL = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL) if md.BERTModel in models else None
base_model = AutoModel.from_pretrained(BERT_MODEL) if md.BERTModel in models else None

def get_best_params( model_name, params_dict, dataset=None, target=None):
    # (This function is unchanged)
    print(f"Fetching best parameters for Model: {model_name}, Target: {target} and Dataset: {dataset}")
    if dataset is not None and target is not None:
        return params_dict[model_name][dataset][target]
    if target is None and dataset is not None:
        return params_dict[model_name][dataset]['overall']
    if target is None and dataset is None:
        return params_dict[model_name]['whole']

# --- MAIN TRAINING AND EVALUATION LOOP ---
for model_class in models:
    model_name = model_class.__name__
    print(f"\n🚀 Training {model_name}...")

    best_params = get_best_params(model_name, best_params_all, dataset=None, target=None)

    if not best_params:
        print(f"❌ No best parameters found for {model_name}, skipping training.")
        continue
    
    # --- Parameter loading (Unchanged) ---
    if model_class in bert_models:
        epochs, batch_size, lr, dropout = best_params["epochs"], best_params["batch_size"], best_params["lr"], best_params["dropout"]
        print(f"🔧 Loaded Best Params for BERT: epochs={epochs}, batch_size={batch_size}, lr={lr}, dropout={dropout}")
    elif model_class in cnn_models:
        epochs, batch_size, lr, dropout, fc1_size = best_params["epochs"], best_params["batch_size"], best_params["lr"], best_params["dropout"], best_params["fc1_size"]
        print(f"🔧 Loaded Best Params for CNN: epochs={epochs}, batch_size={batch_size}, lr={lr}, dropout={dropout}, fc1_size={fc1_size}")
    else: # RNN models
        epochs, batch_size, lr, dropout, hidden_size, num_layers = best_params["epochs"], best_params["batch_size"], best_params["lr"], best_params["dropout"], best_params["hidden_size"], best_params["num_layers"]
        print(f"🔧 Loaded Best Params for RNN: epochs={epochs}, batch_size={batch_size}, lr={lr}, dropout={dropout}, hidden_size={hidden_size}, num_layers={num_layers}")

    # --- Data and Model Initialization (Unchanged) ---
    train_loader, val_loader = get_dataloader(model_class, batch_size) # We ignore the test loader from this function now

    if model_class in cnn_models:
        model = model_class(num_classes, dropout, fc1_size).to(device)
    elif model_class in bert_models:
        base_model = AutoModel.from_pretrained("bert-base-uncased")
        model = model_class(base_model, num_classes).to(device)
    else:
        model = model_class(num_classes, hidden_size, dropout, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)

    # --- Training (Unchanged) ---
    if model_class in bert_models:
        history = train_bert(model, train_loader, val_loader, loss_function, lr, epochs, num_classes, patience=None)
    else:
        history = train(model, train_loader, optimizer, device, epochs, loss_function, num_classes, val_loader=val_loader, patience=None)

    # --- MODIFIED: DUAL EVALUATION BLOCK ---
    
    # 1. Evaluate on WTWT Test Set
    print(f"🧪 Evaluating {model_name} on WTWT test set...")
    if model_class in bert_models:
        wtwt_test_loader = create_transformer_loaders(df_wtwt_test_processed, batch_size, bert_tokenizer, shuffle=False)
        wtwt_true_labels = df_wtwt_test_processed['stance'].values
        wtwt_predictions = get_text_predictions(model, wtwt_test_loader, num_classes)
        wtwt_report = classification_report(wtwt_true_labels, wtwt_predictions, digits=4, labels=interested_labels, target_names=label_names, output_dict=True, zero_division=0)
        f1_score_wtwt = wtwt_report['macro avg']['f1-score']
    else:
        wtwt_test_loader = create_precomputed_loaders(wtwt_test_embeddings, wtwt_test_labels, batch_size=batch_size, shuffle=False)
        f1_score_wtwt, wtwt_predictions, wtwt_true_labels, _ = evaluate_model(model, wtwt_test_loader, device, loss_function, num_classes, interested_labels, label_names)
    
    all_wtwt_predictions[model_class] = wtwt_predictions
    print(f"✅ {model_name} WTWT F1 Score: {f1_score_wtwt:.4f}")

    # 2. Evaluate on Non-WTWT Test Set
    print(f"🧪 Evaluating {model_name} on non-WTWT test set...")
    if model_class in bert_models:
        except_wtwt_test_loader = create_transformer_loaders(df_except_wtwt_test, batch_size, bert_tokenizer, shuffle=False)
        except_wtwt_true_labels = df_except_wtwt_test['stance'].values
        except_wtwt_predictions = get_text_predictions(model, except_wtwt_test_loader, num_classes)
        except_wtwt_report = classification_report(except_wtwt_true_labels, except_wtwt_predictions, digits=4, labels=interested_labels, target_names=label_names, output_dict=True, zero_division=0)
        f1_score_except_wtwt = except_wtwt_report['macro avg']['f1-score']
    else:
        except_wtwt_test_loader = create_precomputed_loaders(except_wtwt_test_embeddings, except_wtwt_test_labels, batch_size=batch_size, shuffle=False)
        f1_score_except_wtwt, except_wtwt_predictions, except_wtwt_true_labels, _ = evaluate_model(model, except_wtwt_test_loader, device, loss_function, num_classes, interested_labels, label_names)

    all_except_wtwt_predictions[model_class] = except_wtwt_predictions
    print(f"✅ {model_name} non-WTWT F1 Score: {f1_score_except_wtwt:.4f}")

    # Store combined results
    result = {
        'model': model_name,
        'f1_score_wtwt': f1_score_wtwt,
        'f1_score_except_wtwt': f1_score_except_wtwt
    }
    results.append(result)

    # Free memory
    del model
    clear_memory()

# --- MODIFIED: FINAL ANALYSIS AND SAVING BLOCK ---

save_dir = RESULTS_DIR / 'evaluation_results'
os.makedirs(save_dir, exist_ok=True)

# Save combined F1 results DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(save_dir, "f1_results.csv"), index=False)
print("\nF1 scores saved to f1_results.csv")

# Helper function for analyzing and saving indices
def analyze_and_save_indices(predictions_dict, true_labels, prefix, save_dir):
    """Analyzes model predictions to find correctly and misclassified examples."""
    print(f"\n🔎 Analyzing results for '{prefix}' test set...")
    
    # Convert true labels to a NumPy array
    true_labels_np = np.array(true_labels)
    
    # Create a NumPy array where each row is a model's predictions
    model_keys = predictions_dict.keys()
    all_predictions_matrix = np.array([predictions_dict[model] for model in model_keys])
    
    if all_predictions_matrix.size == 0:
        print(f"⚠️ No predictions found for '{prefix}'. Skipping analysis.")
        return
        
    # 1. Identify misclassified examples (where NO model predicted correctly)
    misclassified_mask = np.all(all_predictions_matrix != true_labels_np, axis=0)
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # 2. Identify correctly classified examples (where ALL models predicted correctly)
    correctly_classified_mask = np.all(all_predictions_matrix == true_labels_np, axis=0)
    correctly_classified_indices = np.where(correctly_classified_mask)[0]

    # Save the indices
    correct_path = os.path.join(save_dir, f"{prefix}_correctly_classified_indices.npy")
    misclassified_path = os.path.join(save_dir, f"{prefix}_misclassified_indices.npy")
    
    np.save(correct_path, correctly_classified_indices)
    np.save(misclassified_path, misclassified_indices)
    
    print(f"✅ Saved correctly classified indices to: {correct_path}")
    print(f"✅ Saved misclassified indices to: {misclassified_path}")

# Run analysis for both test sets
analyze_and_save_indices(all_wtwt_predictions, df_wtwt_test_processed['stance'].values, 'wtwt', save_dir)
analyze_and_save_indices(all_except_wtwt_predictions, df_except_wtwt_test['stance'].values, 'except_wtwt', save_dir)