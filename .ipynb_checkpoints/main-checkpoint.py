import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
import nltk
import re
from mappings import targets, semeval_labels, wtwt_labels
from sklearn.model_selection import train_test_split
import string
from nltk import word_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import baselines.all_baselines as md
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

target_mappings = {
    "semeval": {
      "at": "Atheism",
      "cc": "Climate Change is a Real Concern",
      "fm": "Feminist Movement",
      "hc": "Hillary Clinton",
      "la": "Legalization of Abortion",
      "dt": "Donald Trump"
  },
  "pstance": {
      "dt": "Donald Trump",
      "ber": "Bernie Sanders",
      "joe": "Joe Biden"
  },
  "covid": {
      "face": "face_masks",
      "fauci": "fauci",
      "school": "school_closures",
      "stay": "stay_at_home_orders"
  }
}


best_params_dict = {'BiGRU': {'whole': {'epochs': 177,
   'batch_size': 128,
   'lr': 0.0007626645379735033,
   'hidden_size': 512,
   'dropout': 0.35766646664263924,
   'num_layers': 3}},
 'ATBiGRU': {'whole': {'epochs': 154,
   'batch_size': 16,
   'lr': 0.0007526842989410973,
   'hidden_size': 512,
   'dropout': 0.22082568946848913,
   'num_layers': 2}},
 'BiLSTM': {'whole': {'epochs': 111,
   'batch_size': 16,
   'lr': 0.00038265840613609846,
   'hidden_size': 512,
   'dropout': 0.2775311919810949,
   'num_layers': 1}},
 'ATBiLSTM': {'whole': {'epochs': 97,
   'batch_size': 16,
   'lr': 0.0006980928452137852,
   'hidden_size': 768,
   'dropout': 0.238696757976767,
   'num_layers': 1}},
 'KimCNN': {'whole': {'epochs': 134,
   'batch_size': 16,
   'lr': 0.0005391647650416826,
   'fc1_size': 256,
   'dropout': 0.12880476191791307}},
 'BERTModel': {'whole': {'epochs': 13,
   'batch_size': 32,
   'lr': 1.6931421144232675e-05,
   'dropout': 0.3157445806245474}}}




df_master = pd.read_csv('dataset/all_combined.csv')
df_master = df_master[df_master['dataset'] != 'wtwt']
semeval_labels_list = ['AGAINST', 'FAVOR', 'NONE']
semeval_labels = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

dataset = None
target_key = None
if target_key is not None:
    target = target_mappings[dataset][source_target_key]
else:
    target = None
    
if dataset == 'pstance':

    interested_labels=[0, 1]
    label_names=['AGAINST', 'FAVOR']
else:
    interested_labels=[0, 1, 2]
    label_names=['AGAINST', 'FAVOR', 'NONE']


print("📥 Loading and filtering training & test datasets...")
df_source_train = filter_df_on_target_dataset_split(target=target, dataset=dataset, df=df_master, split='train')
df_source_test = filter_df_on_target_dataset_split(target=target, dataset=dataset, df=df_master, split='test')

print(f"✅ Training examples: {len(df_source_train)}")
print(f"✅ Test examples: {len(df_source_test)}")

print("🔄 Mapping stance labels...")
df_source_train = map_stance_labels(df_source_train)
df_source_test = map_stance_labels(df_source_test)





df_train, df_val, df_test = stratified_train_val_test_split(df_source_train, df_source_test)

print("\n📊 Label Distribution in Train, Validation, and Test Sets:")
print("----------------------------------------------------")
print(f"🔹 Train Distribution:\n{df_train['stance'].value_counts(normalize=True) * 100}\n")
print(f"🔹 Validation Distribution:\n{df_val['stance'].value_counts(normalize=True) * 100}\n")
print(f"🔹 Test Distribution:\n{df_test['stance'].value_counts(normalize=True) * 100}\n")


SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
sentence_transformer_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER)
sentence_transformer_bert_model = AutoModel.from_pretrained(SENTENCE_TRANSFORMER)
train_embeddings, train_labels = compute_bert_embeddings(df_train, sentence_transformer_tokenizer, sentence_transformer_bert_model)
val_embeddings, val_labels = compute_bert_embeddings(df_val, sentence_transformer_tokenizer, sentence_transformer_bert_model)
test_embeddings, test_labels = compute_bert_embeddings(df_test, sentence_transformer_tokenizer, sentence_transformer_bert_model)


num_classes = df_train['stance'].nunique()
print("Here is num_classes", num_classes)
stance_counts = df_train['stance'].value_counts().sort_index()  # Get class counts


class_weights = 1.0 / stance_counts
class_weights = class_weights / class_weights.sum()  # Normalize weights (optional)
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)

# Choose appropriate loss function
if num_classes == 2:
    loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor[1]).to(device)
else:
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    
    
rnn_models = [md.BiGRU, md.ATBiGRU, md.BiLSTM, md.ATBiLSTM]
cnn_models = [md.KimCNN]
bert_models = [md.BERTModel]
models = rnn_models + cnn_models + bert_models
BERT_MODEL = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL) if md.BERTModel in models else None
base_model = AutoModel.from_pretrained(BERT_MODEL) if md.BERTModel in models else None

def get_dataloader(model_class, batch_size):
    if model_class in bert_models:
        return (
            create_transformer_loaders(df_train, batch_size, bert_tokenizer, shuffle=True),
            create_transformer_loaders(df_val, batch_size, bert_tokenizer, shuffle=False),
            create_transformer_loaders(df_test, batch_size, bert_tokenizer, shuffle=False),
        )
    else:
        return (
            create_precomputed_loaders(train_embeddings, train_labels, batch_size=batch_size, shuffle=True),
            create_precomputed_loaders(val_embeddings, val_labels, batch_size=batch_size, shuffle=False),
            create_precomputed_loaders(test_embeddings, test_labels, batch_size=batch_size, shuffle=False),
        )
    
def initialize_model(model_class):
    if model_class in cnn_models:
        return model_class(num_classes, dropout, fc1_size).to(device)
    elif model_class in bert_models:
        return model_class(base_model, num_classes).to(device)
    else:
        return model_class(num_classes, hidden_size, dropout, num_layers).to(device)
    
results = []  
all_model_predictions = {}
for model_class in models:
    model_name = model_class.__name__
    print(f"\n🚀 Training {model_name}...")
    model_params = best_params_dict.get(model_name, {}).get("whole", {})
    if not model_params:
        print(f"❌ No best parameters found for {model_name}, skipping training.")
        continue
        
    #epochs = model_params.get("epochs", 50)
    epochs = 5# Defaulting to 50 if not found
    batch_size = model_params.get("batch_size", 32)
    lr = model_params.get("lr", 1e-4)
    dropout = model_params.get("dropout", 0.3)

    # Additional params for RNNs & CNNs
    hidden_size = model_params.get("hidden_size", 512)
    num_layers = model_params.get("num_layers", 2)
    fc1_size = model_params.get("fc1_size", 256)  # CNN-specific

    print(f"🔧 Loaded Best Params: {model_params}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloader(model_class, batch_size)

    # Initialize Model
    if model_class in cnn_models:
        model = model_class(num_classes, dropout, fc1_size).to(device)
    elif model_class in bert_models:
        base_model = AutoModel.from_pretrained("bert-base-uncased")
        model = model_class(base_model, num_classes).to(device)
    else:
        model = model_class(num_classes, hidden_size, dropout, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)

    # Train Model
    if model_class in bert_models:
        history = train_bert(model, train_loader, val_loader, loss_function, lr, epochs, num_classes, patience=None)
    else:
        history = train(model, train_loader, optimizer, device, epochs, loss_function, num_classes, val_loader=val_loader, patience=None)

    # Evaluate Model
    if model_class in bert_models:
        true_labels = df_test['stance'].values
        predictions = get_text_predictions(model, test_loader, num_classes)
        report = classification_report(true_labels, predictions, digits=4, labels=interested_labels, target_names=label_names, output_dict=True, zero_division=0)
        f1_score = report['macro avg']['f1-score']
    else:
        f1_score, predictions, true_labels, indices = evaluate_model(model, test_loader, device, loss_function, num_classes, interested_labels, label_names)

    print(f"✅ {model_name} F1 Score: {f1_score:.4f}")
    
    result = {
        'model': model.__class__.__name__,
        'f1_score': f1_score

    }
    results.append(result)
    all_model_predictions[model_class] = predictions
    # Save best model

    # Free memory
    del model
    clear_memory()
    
    



def get_save_directory(dataset=None, target=None):
    """Determines the appropriate save directory based on dataset and target."""
    base_dir = "results"
    
    if dataset is None and target is None:
        return os.path.join(base_dir, "whole")
    elif dataset is not None and target is None:
        return os.path.join(base_dir, dataset, "overall")
    else:
        return os.path.join(base_dir, dataset, target)

    
print(results)
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(save_dir, f"results.csv", index=False)

# Label mapping
label_mapping = {0: 'AGAINST', 1: 'FAVOR', 2: 'NONE'}

# Convert true labels to a NumPy array
true_labels = np.array(true_labels)

# Create a NumPy array where each row is a model's predictions
all_predictions_matrix = np.array([all_model_predictions[model] for model in all_model_predictions])  
# Shape: (num_models, num_samples)

# 1️⃣ Identify misclassified examples (where no model predicted correctly)
misclassified_indices = np.where(np.all(all_predictions_matrix != true_labels, axis=0))[0]

# 2️⃣ Identify correctly classified examples (where all models predicted correctly)
correctly_classified_indices = np.where(np.all(all_predictions_matrix == true_labels, axis=0))[0]

# Get the appropriate save directory
save_dir = get_save_directory(dataset, target)
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Extract misclassified examples
misclassified_examples = df_test.iloc[misclassified_indices][['target', 'text', 'stance']]
correctly_classified_examples = df_test.iloc[correctly_classified_indices][['target', 'text', 'stance']]

# ✅ Save misclassified and correctly classified examples
misclassified_examples.to_csv(os.path.join(save_dir, "misclassified_examples.csv"), index=False)
correctly_classified_examples.to_csv(os.path.join(save_dir, "correctly_classified_examples.csv"), index=False)

print(f"🔍 Found {len(misclassified_examples)} examples misclassified by all models.")
print(f"✅ Found {len(correctly_classified_examples)} examples correctly classified by all models.")

# 3️⃣ Per-label analysis (Save separate CSVs for each label)
for label_value, label_name in label_mapping.items():
    # Find indices of examples with this label
    label_indices = np.where(true_labels == label_value)[0]

    # Get misclassified examples for this label
    misclassified_label_indices = np.intersect1d(misclassified_indices, label_indices)
    misclassified_label_examples = df_test.iloc[misclassified_label_indices][['target', 'text', 'stance']]
    misclassified_label_examples.to_csv(os.path.join(save_dir, f"misclassified_examples_{label_name}.csv"), index=False)

    # Get correctly classified examples for this label
    correctly_classified_label_indices = np.intersect1d(correctly_classified_indices, label_indices)
    correctly_classified_label_examples = df_test.iloc[correctly_classified_label_indices][['target', 'text', 'stance']]
    correctly_classified_label_examples.to_csv(os.path.join(save_dir, f"correctly_classified_examples_{label_name}.csv"), index=False)

    print(f"📁 Saved {len(misclassified_label_examples)} misclassified examples for label: {label_name}")
    print(f"📁 Saved {len(correctly_classified_label_examples)} correctly classified examples for label: {label_name}")

print("\n📊 All per-label classification files saved successfully!")