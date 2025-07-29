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


def objective(trial, model_class):
    """
    Objective function for Optuna hyperparameter tuning.
    """

    # **🔹 Define hyperparameters to tune**
    model_name = model_class.__name__
    print('*'*80, model_name)
    if model_class in cnn_models:
        epochs = trial.suggest_int("epochs", 25, 200)
        #epochs = trial.suggest_int("epochs", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        fc1_size = trial.suggest_categorical("fc1_size", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        print(f"🔍 Trial {trial.number}: epochs={epochs}, batch_size={batch_size}, lr={lr}, fc1_size={fc1_size}, dropout={dropout}")
    elif model_class in bert_models:
        epochs = trial.suggest_int("epochs", 3, 10)  # BERT fine-tuning is data-intensive, fewer epochs often work well
        #epochs = trial.suggest_int("epochs", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])  # GPU memory constraints limit large batches
        lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)  # BERT models typically require lower learning rates
        dropout = trial.suggest_float("dropout", 0.1, 0.5)  # Regularization to prevent overfitting
    else:
        epochs = trial.suggest_int("epochs", 25, 200)
        #epochs = trial.suggest_int("epochs", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 768])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        num_layers = trial.suggest_int("num_layers", 1, 3)

        print(f"🔍 Trial {trial.number}: epochs={epochs}, batch_size={batch_size}, lr={lr}, hidden_size={hidden_size}, dropout={dropout}, num_layers={num_layers}")

    if model_class not in bert_models:  
        train_loader = create_precomputed_loaders(train_embeddings, train_labels, batch_size=batch_size, shuffle=True)
        val_loader = create_precomputed_loaders(val_embeddings, val_labels, batch_size=batch_size, shuffle=False)
        test_loader = create_precomputed_loaders(test_embeddings, test_labels, batch_size=batch_size, shuffle=False) 
    else:
        BERT_MODEL = "bert-base-uncased"
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        train_loader_bert = create_transformer_loaders(df_train, batch_size, bert_tokenizer, shuffle=True)
        val_loader_bert = create_transformer_loaders(df_val, batch_size, bert_tokenizer, shuffle=False)
        test_loader_bert = create_transformer_loaders(df_test, batch_size, bert_tokenizer, shuffle=False)       
    
    # **🔹 Initialize model**
    if model_class in cnn_models:
        model = model_class(num_classes, dropout, fc1_size).to(device)
    elif model_class in bert_models:
        base_model = AutoModel.from_pretrained(BERT_MODEL)
        model = model_class(base_model, num_classes).to(device)
    else:
        model = model_class(num_classes, hidden_size, dropout, num_layers).to(device)

    # **🔹 Optimizer**
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)

    # **🔹 Train model**
    if model_class in bert_models:
        history = train_bert(model, train_loader_bert, val_loader_bert, loss_function, lr, epochs, num_classes, patience = None)
    else:
        history = train(model, train_loader, optimizer, device, epochs, loss_function, num_classes, val_loader=val_loader, patience=None)

    # **🔹 Evaluate model**
    if model_class in bert_models:
        true_labels = df_test['stance'].values
        predictions = get_text_predictions(model, test_loader_bert, num_classes)
        report = classification_report(true_labels, predictions, digits=4, labels=interested_labels, target_names=label_names, output_dict=True, zero_division=0)
        f1_score = report['macro avg']['f1-score']
        print("Here is F1 Score", f1_score)
    else:
        f1_score, predictions, true_labels, indices = evaluate_model(model, test_loader, device, loss_function, num_classes, interested_labels, label_names)

    # **🔹 Save best model**
    trial.set_user_attr("history", history)

    # **🔹 Clear memory**
    del model
    clear_memory()

    return f1_score 


def run_optuna_for_all_models(models, save_base_dir, target, dataset, n_trials=100, n_jobs=1):
    """
    Runs Optuna tuning for all RNN models and saves results.
    """
    for model_class in models:
        model_name = model_class.__name__
        print(f"\n🚀 Starting Optuna tuning for: {model_name}")
        
        if target is not None:
            save_dir = os.path.join(save_base_dir, model_name, dataset, target)
        else:
            save_dir = os.path.join(save_base_dir, model_name, dataset, 'overall' )
        os.makedirs(save_dir, exist_ok=True)

        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_class), n_trials=n_trials, n_jobs=n_jobs)

        # ✅ Save best parameters
        best_trial = {
            "best_trial_number": study.best_trial.number,
            "best_f1_score": study.best_value,
            "best_parameters": study.best_params
        }

        with open(os.path.join(save_dir, "best_params.json"), "w") as f:
            json.dump(best_trial, f, indent=4)
        print(f"✅ Best parameters saved for {model_name}")

        # ✅ Save training history
        history = {
            "train_loss": study.trials_dataframe()["value"].tolist(),
            "params": [trial.params for trial in study.trials]
        }

        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)

        # ✅ Save Optuna plots
        plots = {
            
            "param_importances": vis.plot_param_importances(study),

        }

        for name, plot in plots.items():
            plot_path = os.path.join(save_dir, f"{name}.png")
            try:
                plot.write_image(plot_path)
                print(f"✅ Saved {name}.png for {model_name}")
            except Exception as e:
                print(f"❌ Failed to save {name}.png for {model_name}: {e}")

        print(f"\n🎯 Fine-tuning completed for {model_name}! Results saved at {save_dir}\n")


# def run_optuna_tuning(n_trials=3, n_jobs=1):
#     """
#     Runs Optuna hyperparameter tuning, saves best parameters, training history, and visualizations.

#     Parameters:
#     - n_trials (int): Number of trials for optimization.
#     - n_jobs (int): Number of parallel jobs (-1 for all available cores).
#     """
    
#     # ✅ Enable parallel execution (n_jobs=-1 for max cores)
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)  

#     # 📌 Check if trials exist before accessing `best_trial`
#     if len(study.trials) == 0 or study.best_trial is None:
#         print("❌ No successful trials. Check your objective function.")
#         return

#     # ✅ Save Best Parameters as JSON
#     best_trial = {
#         "best_trial_number": study.best_trial.number,
#         "best_f1_score": study.best_value,
#         "best_parameters": study.best_params
#     }

#     best_params_path = os.path.join(save_dir, "best_params.json")
#     with open(best_params_path, "w") as f:
#         json.dump(best_trial, f, indent=4)
#     print(f"✅ Saved best parameters to {best_params_path}")

#     # ✅ Save Training History
#     history = {
#         "train_loss": study.trials_dataframe()["value"].tolist(),
#         "params": [trial.params for trial in study.trials]
#     }

#     history_path = os.path.join(save_dir, "training_history.json")
#     with open(history_path, "w") as f:
#         json.dump(history, f, indent=4)
#     print(f"✅ Saved training history to {history_path}")

#     # 📊 Plot and Save All Optuna Visualizations
#     plots = {
#         "optimization_history": vis.plot_optimization_history(study),
#         "param_importances": vis.plot_param_importances(study),
#         "slice": vis.plot_slice(study),
#         "parallel_coordinates": vis.plot_parallel_coordinate(study),
#         "contour": vis.plot_contour(study)
#     }

#     for name, plot in plots.items():
#         plot_path = os.path.join(save_dir, f"{name}.png")
#         try:
#             plot.write_image(plot_path)  # ✅ Ensure plotly + kaleido are installed
#             print(f"✅ Saved {name}.png at {save_dir}")
#         except Exception as e:
#             print(f"❌ Failed to save {name}.png: {e}")

#     print("\n🎯 Fine-tuning complete! Best parameters and results saved.")

rnn_models = [md.BiGRU, md.ATBiGRU, md.BiLSTM, md.ATBiLSTM]
cnn_models = [md.KimCNN]
bert_models = [md.BERTModel]
training_non_bert = True


df_master = pd.read_csv('dataset/all_combined.csv')
df_master = df_master[df_master['dataset'] != 'wtwt']
semeval_labels_list = ['AGAINST', 'FAVOR', 'NONE']
semeval_labels = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

for dataset in target_mappings:
    if dataset == 'pstance':


        target = None
        print(dataset, target)

        if dataset == 'pstance':
            interested_labels=[0, 1]
            label_names=['AGAINST', 'FAVOR']

        else:
            interested_labels=[0, 1, 2]
            label_names=['AGAINST', 'FAVOR', 'NONE']

        print(f"🔹 Using dataset: {dataset} for target: {target}")

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

        if training_non_bert:
            SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
            sentence_transformer_tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER)
            sentence_transformer_bert_model = AutoModel.from_pretrained(SENTENCE_TRANSFORMER)
            train_embeddings, train_labels = compute_bert_embeddings_token(df_train, sentence_transformer_tokenizer, sentence_transformer_bert_model)
            val_embeddings, val_labels = compute_bert_embeddings_token(df_val, sentence_transformer_tokenizer, sentence_transformer_bert_model)
            test_embeddings, test_labels = compute_bert_embeddings_token(df_test, sentence_transformer_tokenizer, sentence_transformer_bert_model)
            #models = [md.BiGRU, md.ATBiGRU, md.BiLSTM, md.ATBiLSTM, md.KimCNN]
            #models = [md.BiLSTM, md.ATBiLSTM, md.KimCNN]
            models = [md.KimCNN]

        else:
            #models = [md.BiGRU, md.ATBiGRU, md.BiLSTM, md.ATBiLSTM, md.KimCNN, md.BERTModel]
            models = [md.BERTModel]


        num_classes = df_train['stance'].nunique()
        print("Here is num_classes", num_classes)
        stance_counts = df_train['stance'].value_counts().sort_index()  # Get class counts

        # Compute class weights
        class_weights = 1.0 / stance_counts
        class_weights = class_weights / class_weights.sum()  # Normalize weights (optional)
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)

        # Choose appropriate loss function
        if num_classes == 2:
            loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor[1]).to(device)
        else:
            loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

        # ✅ Run tuning for all models


        save_base_dir = f"/home/p/parush/style_markers/fine_tuning/"
        #dataset = 'whole' # Only uncommnet this when training on all datasets togeather
        run_optuna_for_all_models(models, save_base_dir, target, dataset, n_trials=200, n_jobs=1)  # 🔥 Runs 30 trials in parallel
        completion_file = os.path.join(save_base_dir, f"{dataset}_{target}_RNN_status.txt")
        with open(completion_file, "w") as f:
            f.write("✅ Training pipeline finished successfully without any errors.")

        print(f"\n🎉 All models trained successfully! Completion file saved at {completion_file}")



