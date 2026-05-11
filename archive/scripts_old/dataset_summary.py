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


df_master = pd.read_csv('dataset/all_combined.csv')
df_master = df_master[df_master['dataset'] != 'wtwt']
semeval_labels_list = ['AGAINST', 'FAVOR', 'NONE']
semeval_labels = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

np.unique(df_master.stance
)


import pandas as pd

def compute_dataset_summary(df):
    # 1) Per‑target & split counts
    counts = (
        df
        .groupby(['dataset','target','type','stance'])
        .size()
        .unstack(fill_value=0)    # columns: FAVOR, AGAINST, NONE
        .reset_index()
    )
    # ensure all stance columns exist
    for col in ['FAVOR','AGAINST','NONE']:
        if col not in counts:
            counts[col] = 0

    # totals & percentages per (dataset, target, split)
    counts['total'] = counts[['FAVOR','AGAINST','NONE']].sum(axis=1)
    counts['favor_pct']   = (counts['FAVOR']   / counts['total'] * 100).round(2)
    counts['against_pct'] = (counts['AGAINST'] / counts['total'] * 100).round(2)
    counts['neutral_pct'] = (counts['NONE']    / counts['total'] * 100).round(2)

    # 2) Pivot train/test into wide columns
    per_target = counts.pivot(
        index=['dataset','target'],
        columns='type',
        values=['favor_pct','against_pct','neutral_pct','total']
    )
    per_target.columns = [f"{split}_{stat}" for stat, split in per_target.columns]
    per_target = per_target.reset_index()

    # 3) Dataset‑level totals (target = "Total")
    ds_counts = (
        df
        .groupby(['dataset','type','stance'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ['FAVOR','AGAINST','NONE']:
        if col not in ds_counts:
            ds_counts[col] = 0
    ds_counts['total'] = ds_counts[['FAVOR','AGAINST','NONE']].sum(axis=1)
    ds_counts['favor_pct']   = (ds_counts['FAVOR']   / ds_counts['total'] * 100).round(2)
    ds_counts['against_pct'] = (ds_counts['AGAINST'] / ds_counts['total'] * 100).round(2)
    ds_counts['neutral_pct'] = (ds_counts['NONE']    / ds_counts['total'] * 100).round(2)

    ds_wide = ds_counts.pivot(
        index='dataset',
        columns='type',
        values=['favor_pct','against_pct','neutral_pct','total']
    )
    ds_wide.columns = [f"{split}_{stat}" for stat, split in ds_wide.columns]
    ds_wide = ds_wide.reset_index().assign(target='Total')

    # 4) Global totals across all datasets (dataset = "All", target = "Total")
    global_counts = (
        df
        .groupby(['type','stance'])
        .size()
        .unstack(fill_value=0)
    )
    # ensure all stance columns exist
    for col in ['FAVOR','AGAINST','NONE']:
        if col not in global_counts.columns:
            global_counts[col] = 0
    # compute totals & pct
    global_counts['total'] = global_counts[['FAVOR','AGAINST','NONE']].sum(axis=1)
    global_counts['favor_pct']   = (global_counts['FAVOR']   / global_counts['total'] * 100).round(2)
    global_counts['against_pct'] = (global_counts['AGAINST'] / global_counts['total'] * 100).round(2)
    global_counts['neutral_pct'] = (global_counts['NONE']    / global_counts['total'] * 100).round(2)

    # pick train vs test row, build one DataFrame row
    gt = global_counts.loc['train']
    gs = global_counts.loc['test']
    global_row = {
        'dataset':        'All',
        'target':         'Total',
        'train_favor_pct':   gt['favor_pct'],
        'train_against_pct': gt['against_pct'],
        'train_neutral_pct': gt['neutral_pct'],
        'train_total':       gt['total'],
        'test_favor_pct':    gs['favor_pct'],
        'test_against_pct':  gs['against_pct'],
        'test_neutral_pct':  gs['neutral_pct'],
        'test_total':        gs['total'],
    }
    global_df = pd.DataFrame([global_row])

    # 5) Concatenate: per-target → per-dataset → global
    summary = pd.concat([per_target, ds_wide, global_df], ignore_index=True, sort=False)

    # 6) Select & order columns
    cols = [
        'dataset','target',
        'train_favor_pct','train_against_pct','train_neutral_pct','train_total',
        'test_favor_pct','test_against_pct','test_neutral_pct','test_total'
    ]
    return summary[cols]

# Example usage:
summary_df = compute_dataset_summary(df_master)
print(summary_df)
summary_df.to_csv("dataset_summary_with_global.csv", index=False)