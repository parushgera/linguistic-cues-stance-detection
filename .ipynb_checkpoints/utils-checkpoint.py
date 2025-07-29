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
import matplotlib.pyplot as plt
from tqdm import tqdm


TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
bert_model = AutoModel.from_pretrained(TOKENIZER_NAME)



def _remove_amp(text):
    return text.replace("&amp;", " ")

def _remove_mentions(text):
    return re.sub(r'(@.*?)[\s]', ' ', text)

def _remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)

def _remove_retweets(text):
    return re.sub(r'^RT[\s]+', ' ', text)

def _remove_links(text):
    return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

def _remove_hashes(text):
    return re.sub(r'#', ' ', text)

def _stitch_text_tokens_together(text_tokens):
    return " ".join(text_tokens)

def _tokenize(text):
    return nltk.word_tokenize(text, language="english")

def _stopword_filtering(text_tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in text_tokens if token not in stop_words]

def _stemming(text_tokens):
    porter = nltk.stem.porter.PorterStemmer()
    return [porter.stem(token) for token in text_tokens]

def _lemmatize_text(text_tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in text_tokens]  
    

def _remove_numbers(text):
    return re.sub(r'\d+', ' ', text)

def _lowercase(text):
    return text.lower()

def _remove_punctuation(text):
    return ''.join(character for character in text if character not in string.punctuation)

def clean_text(text):
    text = _remove_amp(text)
    text = _remove_links(text)
    text = _remove_hashes(text)
    text = _remove_retweets(text)
    text = _remove_mentions(text)
    text = _remove_multiple_spaces(text)
    text = _lowercase(text)
    return text.strip()


def filter_df_on_target_dataset_split(target, dataset, split, df):
    # Filter on both 'target' and 'dataset' columns
    filtered_df = df[(df['target'] == target) & (df['dataset'] == dataset) & (df['type'] == split)].copy()
    
    # Drop 'type' and 'dataset' columns
    filtered_df = filtered_df.drop(columns=['type', 'dataset'])
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df


def stratified_train_val_test_split(df_source_train, df_source_test, val_size=0.1, random_state=42):


    # Split df_source_train into train and validation sets using stratification on 'stance'
    df_train, df_val = train_test_split(
        df_source_train, 
        test_size=val_size, 
        stratify=df_source_train['stance'], 
        random_state=random_state
    )

    # df_source_test remains as the test set
    df_test = df_source_test.copy()

    return df_train, df_val, df_test

def map_stance_labels(df):
    df = df.copy()  # Ensure original DataFrame is not modified
    df['stance'] = df['stance'].map(semeval_labels)
    return df



def load_glove(embedding_dim, source):
    embeddings_dictionary = dict()
    if source == 'twitter':
        glove_file = open(f'/data/parush/embeddings/twitter/glove.twitter.27B.{embedding_dim}d.txt', encoding="utf8")
    else:
        glove_file = open(f'/data/parush/embeddings/wikipedia/glove.6B.{embedding_dim}d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    return embeddings_dictionary

def make_vocab(data_list, source_target):
    word_counter = Counter()
    for text in data_list:
        words = word_tokenize(text.lower())
        word_counter.update(words)
    words_in_target = word_tokenize(source_target.lower())
    word_counter.update(words_in_target)
    return word_counter
def clear_memory():
    # Clears CUDA memory cache and invokes garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
def index_vocab(word_counter):
    word2index = {'unk' :0}
    for i, word in enumerate(word_counter.keys()):
        word2index[word] = i + 1
    index2word = {v :k for k, v in word2index.items()}
    return word2index, index2word

def get_embedding_matrix(word_counter, embedding_dim, word2index, embeddings_dictionary):
    embedding_matrix = torch.zeros((len(word_counter ) +1, embedding_dim))
    for word, index in word2index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = torch.from_numpy(embedding_vector)
    return embedding_matrix

def texts2tensor(texts, word2index, pad_token=0, max_len=50):
    # print(texts)
    indexes_list = [[word2index.get(word, 0) for word in word_tokenize(text)] for text in texts]
    truncated_indexes = [indexes[:max_len] for indexes in indexes_list]
    padded_indexes = [indexes + [0] * (max_len - len(indexes)) for indexes in truncated_indexes]
    return torch.LongTensor(padded_indexes)





# class TweetDataset(Dataset):
#     def __init__(self, dataframe, source_target, word2index, max_seq_len=None):
#         texts = dataframe.text.values.tolist()

#         self.texts = [self._preprocess(text) for text in texts]

#         #self._print_random_samples(self.texts)

#         self.word2index = word2index

#         if 'stance' in dataframe:
#             classes = dataframe.stance.values.tolist()
#             self.labels = classes

#         self.target_word = source_target
#         # target_word = source_target
#         # self.target_word = [word for word in target_word]
#         self.max_seq_len = max([len(text.split()) for text in self.texts])
#         if max_seq_len is None:
#             self.max_seq_len = max([len(text.split()) for text in self.texts])
#         else:
#             self.max_seq_len = max_seq_len

# #     def _print_random_samples(self, texts):
# #         np.random.seed(42)
# #         random_entries = np.random.randint(0, len(texts), 5)

# #         for i in random_entries:
# #             print(f"Entry {i}: {texts[i]}")

# #         print()

#     def _preprocess(self, text):
#         text = self._remove_amp(text)
#         text = self._remove_links(text)
#         # text = self._remove_hashes(text)
#         text = self._remove_retweets(text)
#         text = self._remove_mentions(text)
#         text = self._remove_multiple_spaces(text)

#         text = self._lowercase(text)
#         text = self._remove_punctuation(text)
#         # text = self._remove_numbers(text)

#         text_tokens = self._tokenize(text)
#         text_tokens = self._stopword_filtering(text_tokens)
#         text_tokens = self._stemming(text_tokens)
#         text = self._stitch_text_tokens_together(text_tokens)

#         return text.strip()


#     def _remove_amp(self, text):
#         return text.replace("&amp;", " ")

#     def _remove_mentions(self, text):
#         return re.sub(r'(@.*?)[\s]', ' ', text)

#     def _remove_multiple_spaces(self, text):
#         return re.sub(r'\s+', ' ', text)

#     def _remove_retweets(self, text):
#         return re.sub(r'^RT[\s]+', ' ', text)

#     def _remove_links(self, text):
#         return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

#     def _remove_hashes(self, text):
#         return re.sub(r'#', ' ', text)

#     def _stitch_text_tokens_together(self, text_tokens):
#         return " ".join(text_tokens)

#     def _tokenize(self, text):
#         return nltk.word_tokenize(text, language="english")

#     def _stopword_filtering(self, text_tokens):
#         stop_words = nltk.corpus.stopwords.words('english')

#         return [token for token in text_tokens if token not in stop_words]

#     def _stemming(self, text_tokens):
#         porter = nltk.stem.porter.PorterStemmer()
#         return [porter.stem(token) for token in text_tokens]

#     def _remove_numbers(self, text):
#         return re.sub(r'\d+', ' ', text)

#     def _lowercase(self, text):
#         return text.lower()

#     def _remove_punctuation(self, text):
#         return ''.join(character for character in text if character not in string.punctuation)

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         # Convert texts and target words to tensors, applying padding
#         text_vec = texts2tensor([self.texts[idx]], self.word2index, max_len=self.max_seq_len)[0]
#         target_word_vec = texts2tensor([self.target_word], self.word2index, max_len=self.max_seq_len)[0]

#         # Ensure both are padded to max_seq_len
#         text_vec = self._pad_sequence(text_vec, self.max_seq_len)
#         target_word_vec = self._pad_sequence(target_word_vec, self.max_seq_len)

#         label = self.labels[idx]
#         return torch.tensor(text_vec, dtype=torch.long), torch.tensor(target_word_vec, dtype=torch.long), torch.tensor \
#             (label, dtype=torch.long)
#     def _pad_sequence(self, sequence, max_len):
#         # Pad or truncate the sequence to max_len
#         if len(sequence) < max_len:
#             # Pad with 0s
#             sequence += [0] * (max_len - len(sequence))
#         else:
#             # Truncate to max_len
#             sequence = sequence[:max_len]
#         return sequence


class TweetDataset(Dataset):
    def __init__(self, dataframe, source_target, word2index, max_seq_len=None):
        self.word2index = word2index
        self.texts = [self._preprocess(text) for text in dataframe.text.values.tolist()]
        
        # Compute max_seq_len if not provided
        if max_seq_len is None:
            self.max_seq_len = max(len(text.split()) for text in self.texts)
        else:
            self.max_seq_len = max_seq_len
        
        # Store labels if available
        self.labels = dataframe.stance.values.tolist() if 'stance' in dataframe else None

        # Store target word
        self.target_word = source_target 

    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)
        text = self._lowercase(text)
        text = self._remove_punctuation(text)
        
        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        text_tokens = self._stemming(text_tokens)
        
        return self._stitch_text_tokens_together(text_tokens)

    def _remove_amp(self, text): return text.replace("&amp;", " ")
    def _remove_mentions(self, text): return re.sub(r'(@.*?)[\s]', ' ', text)
    def _remove_multiple_spaces(self, text): return re.sub(r'\s+', ' ', text)
    def _remove_retweets(self, text): return re.sub(r'^RT[\s]+', ' ', text)
    def _remove_links(self, text): return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)
    def _remove_punctuation(self, text): return ''.join(c for c in text if c not in string.punctuation)
    def _lowercase(self, text): return text.lower()
    def _tokenize(self, text): return nltk.word_tokenize(text, language="english")
    def _stitch_text_tokens_together(self, text_tokens): return " ".join(text_tokens)
    def _stopword_filtering(self, text_tokens):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return [token for token in text_tokens if token not in stop_words]
    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Convert text to tensor
        text_vec = texts2tensor([self.texts[idx]], self.word2index, max_len=self.max_seq_len)[0]

        # ✅ FIXED: Tokenize multi-word target correctly
        target_tokens = nltk.word_tokenize(self.target_word)  # Tokenize multi-word target
        target_word_vec = texts2tensor([" ".join(target_tokens)], self.word2index, max_len=self.max_seq_len)[0]

        # Ensure proper padding
        text_vec = self._pad_sequence(text_vec, self.max_seq_len)
        target_word_vec = self._pad_sequence(target_word_vec, self.max_seq_len)

        if self.labels is not None:
            label = self.labels[idx]
            return (
                torch.tensor(text_vec, dtype=torch.long),
                torch.tensor(target_word_vec, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
            )
        else:
            return (
                torch.tensor(text_vec, dtype=torch.long),
                torch.tensor(target_word_vec, dtype=torch.long),
            )
    def _pad_sequence(self, sequence, max_len):
        """Pads or truncates a sequence to max_len."""
        sequence = sequence[:max_len]  # Truncate if longer
        padding = torch.zeros(max_len - sequence.shape[0], dtype=torch.long)  # Create tensor padding
        return torch.cat([sequence, padding])  # Concatenate as tensors

    
class TweetDatasetRoberta(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        
        self.max_length = max_length
        texts = dataframe.text.values.tolist()
        texts = [self._preprocess(text) for text in texts]

        #self._print_random_samples(texts)

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        if 'stance' in dataframe:
            classes = dataframe.stance.values.tolist()
            self.labels = classes

    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        #text = self._lowercase(text)
        text = self._remove_punctuation(text)
        #text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        #text_tokens = self._stemming(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()


    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)
    
    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)

    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in text_tokens if token not in stop_words]

    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    def _remove_numbers(self, text):
        return re.sub(r'\d+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label

def get_text_predictions(model, loader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)


    predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)


            outputs = model(input_ids, attention_mask)

            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions






def create_rnn_loaders(df, target, word2index, batch_size, shuffle):
    dataset = TweetDataset(df, target, word2index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader






# def train(model, train_loader, optimizer, device, num_epochs, loss_function, val_loader=None, patience=None):
#     """
#     Trains the model while tracking loss history for plotting.
    
#     Returns:
#     - history: Dictionary containing lists of training and validation loss per epoch.
#     """
#     history = {'train_loss': [], 'val_loss': []}  # Store loss per epoch
#     best_val_loss = float('inf')  
#     no_improve_epochs = 0  
#     best_model_state = None 

#     for epoch in range(num_epochs):
#         model.train()  
#         running_loss = 0.0  

#         for batch_idx, (text_emb, label) in enumerate(train_loader):
#             text_emb, label = text_emb.to(device), label.to(device)
#             optimizer.zero_grad()
#             output = model(text_emb)  # Pass BERT embeddings
#             loss = loss_function(output, label)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)  # Gradient Clipping
#             optimizer.step()
#             running_loss += loss.item()

#             if batch_idx % 100 == 0:
#                 print(f'Epoch: {epoch + 1}, Batch: {batch_idx}, Training Loss: {loss.item():.6f}')
        
#         avg_train_loss = running_loss / len(train_loader)
#         history['train_loss'].append(avg_train_loss)
#         print(f'Epoch: {epoch + 1}, Average Training Loss: {avg_train_loss:.6f}')

#         if val_loader is not None:
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for text_emb, label in val_loader:
#                     text_emb, label = text_emb.to(device), label.to(device)
#                     output = model(text_emb)
#                     loss = loss_function(output, label)
#                     val_loss += loss.item()

#             avg_val_loss = val_loss / len(val_loader)
#             history['val_loss'].append(avg_val_loss)
#             print(f'Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.6f}')

#             if patience:
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     no_improve_epochs = 0  # Reset patience counter
#                     best_model_state = model.state_dict()  # Save best model
#                 else:
#                     no_improve_epochs += 1
#                     print(f"No improvement in validation loss for {no_improve_epochs}/{patience} epochs")

#                 if no_improve_epochs >= patience:
#                     print(f"\nEarly stopping triggered after {epoch+1} epochs!")
#                     break  # Stop training if patience limit is reached
        
#     print('Training complete')
#     return history  # ✅ Return loss history for plotting
    


    
    
    
    
    
    
    
    
    
    
    


def train(model, train_loader, optimizer, device, num_epochs, loss_function, val_loader=None, patience=None):
    """
    Trains the model while tracking loss history for plotting.
    
    Returns:
    - history: Dictionary containing lists of training and validation loss per epoch.
    """
    history = {'train_loss': [], 'val_loss': []}  # Store loss per epoch
    best_val_loss = float('inf')  
    no_improve_epochs = 0  
    best_model_state = None 

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0  

        # ✅ Add tqdm progress bar for training loop
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for batch_idx, (text_emb, label) in train_pbar:
            text_emb, label = text_emb.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text_emb)  # Pass BERT embeddings
            loss = loss_function(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)  # Gradient Clipping
            optimizer.step()
            running_loss += loss.item()

            # ✅ Update tqdm progress bar
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs} -> Average Training Loss: {avg_train_loss:.6f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            # ✅ Add tqdm progress bar for validation loop
            val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
            with torch.no_grad():
                for text_emb, label in val_pbar:
                    text_emb, label = text_emb.to(device), label.to(device)
                    output = model(text_emb)
                    loss = loss_function(output, label)
                    val_loss += loss.item()

                    # ✅ Update tqdm progress bar
                    val_pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} -> Validation Loss: {avg_val_loss:.6f}")

            # ✅ Implement early stopping
            if patience:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_epochs = 0  # Reset patience counter
                    best_model_state = model.state_dict()  # Save best model
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in validation loss for {no_improve_epochs}/{patience} epochs")

                if no_improve_epochs >= patience:
                    print(f"\n🚨 Early stopping triggered after {epoch+1} epochs!")
                    break  # Stop training if patience limit is reached

    print("✅ Training complete")
    return history  # ✅ Return loss history for plotting
    
    
    
    
def plot_loss(history, model_class, save_path=None):

    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')

    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Validation Loss', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    if save_path is None:
        save_path = Path(os.getcwd())  # Default to current directory
    else:
        save_path = Path(save_path)  # Ensure it's a Path object

    # Save the figure
    output_file = save_path / f"{model_class}_training_loss_plot.png"
    plt.savefig(output_file, bbox_inches='tight')  # Save with tight bounding box
    print(f"Plot saved at: {output_file}")
    #plt.show()
# def evaluate_model(model, test_loader, device, loss_function, interested_labels, label_names):
#     model.eval()  # Set the model to evaluation mode
#     total_test_loss = 0
#     all_predictions = []
#     all_true_labels = []
#     with torch.no_grad():
#         for data, target_word, label in test_loader:
#             data, target_word, label = data.to(device), target_word.to(device), label.to(device)
#             output = model(data, [len(txt) for txt in data], None, target_word)
#             loss = loss_function(output, label)
#             total_test_loss += loss.item()
#             _, predicted = torch.max(output, 1)
            
#             all_predictions.extend(predicted.cpu().numpy())
#             all_true_labels.extend(label.cpu().numpy())
#     avg_test_loss = total_test_loss / len(test_loader)
#     print("here is all_prediction",all_predictions )
#     print("here is all_prediction",all_true_labels)
#     report = classification_report(all_true_labels, all_predictions, labels=interested_labels, target_names=label_names, output_dict=True)
#     return report['macro avg']['f1-score']

# def evaluate_model(model, test_loader, device, loss_function, interested_labels, label_names):
#     model.eval()  # Set model to evaluation mode
#     total_test_loss = 0
#     all_predictions = []
#     all_true_labels = []
#     all_indices = []  # To track example indices

#     with torch.no_grad():
#         for batch_idx, (data, target_word, label) in enumerate(test_loader):
#             data, target_word, label = data.to(device), target_word.to(device), label.to(device)
#             output = model(data, [len(txt) for txt in data], None, target_word)
#             loss = loss_function(output, label)
#             total_test_loss += loss.item()
#             _, predicted = torch.max(output, 1)

#             all_predictions.extend(predicted.cpu().numpy())
#             all_true_labels.extend(label.cpu().numpy())
#             all_indices.extend(range(batch_idx * test_loader.batch_size, batch_idx * test_loader.batch_size + len(label)))

#     avg_test_loss = total_test_loss / len(test_loader)
#     report = classification_report(all_true_labels, all_predictions, labels=interested_labels, target_names=label_names, output_dict=True)
#     f1_score = report['macro avg']['f1-score']

#     return f1_score, all_predictions, all_true_labels, all_indices

# from sklearn.metrics import classification_report
# import torch

def evaluate_model(model, test_loader, device, loss_function, interested_labels, label_names):
 
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0
    all_predictions = []
    all_true_labels = []
    all_indices = []  # To track example indices

    with torch.no_grad():
        for batch_idx, (text_emb, label) in enumerate(test_loader):  
            text_emb, label = text_emb.to(device), label.to(device)  # ✅ Move to device

            output = model(text_emb)  # ✅ Forward pass (no target_word)
            loss = loss_function(output, label)
            total_test_loss += loss.item()

            _, predicted = torch.max(output, 1)  # Get class predictions

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(label.cpu().numpy())
            all_indices.extend(range(batch_idx * test_loader.batch_size, batch_idx * test_loader.batch_size + len(label)))

    avg_test_loss = total_test_loss / len(test_loader)

    # ✅ Generate classification report
    report = classification_report(all_true_labels, all_predictions, labels=interested_labels, 
                                   target_names=label_names, output_dict=True)
    f1_score = report['macro avg']['f1-score']

    print(f"Test Loss: {avg_test_loss:.6f}, Macro F1-Score: {f1_score:.4f}")

    return f1_score, all_predictions, all_true_labels, all_indices


def train_roberta(model, train_dataloader, val_dataloader, learning_rate, epochs, patience = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    no_improve_epochs = 0  # Counter for early stopping
    best_model_state = None  # Store the best model state


    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()  # Set the model to training mode

        for train_input, train_label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training") :
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label)
            total_loss_train += loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()

        # Validation phase
        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            model.eval()  # Set the model to evaluation mode

            for val_input, val_label in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)
                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)
                loss = criterion(output, val_label)
                total_loss_val += loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
            avg_train_loss = total_loss_train / len(train_dataloader)
            avg_train_acc = total_acc_train / len(train_dataloader.dataset)
            avg_val_loss = total_loss_val / len(val_dataloader)
            avg_val_acc = total_acc_val / len(val_dataloader.dataset)

            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(avg_val_acc)

            print(f'\nEpoch {epoch + 1}/{epochs} '
                  f'| Train Loss: {avg_train_loss:.3f} '
                  f'| Train Accuracy: {avg_train_acc:.3f} '
                  f'| Val Loss: {avg_val_loss:.3f} '
                  f'| Val Accuracy: {avg_val_acc:.3f}\n')

            # Early Stopping Logic
            if patience:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_epochs = 0  # Reset counter
                    best_model_state = model.state_dict()  # Save best model
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in validation loss for {no_improve_epochs}/{patience} epochs")

                if no_improve_epochs >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    break  #
                


    return history


def prepare_data(df_train, df_val, df_test, tokenizer, max_length=128):

    df_train = df_train.drop(columns = ['target'])
    df_val = df_val.drop(columns = ['target'])
    df_test = df_test.drop(columns = ['target'])
    num_classes = len(np.unique(df_train.stance.values))

    train_dataloader = DataLoader(TweetDatasetRoberta(df_train, tokenizer, max_length), batch_size=64, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(TweetDatasetRoberta(df_val, tokenizer, max_length), batch_size=64, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(TweetDatasetRoberta(df_test, tokenizer, max_length), 
                                 batch_size=64, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader, num_classes




class TweetDatasetBertEmbeddings(Dataset):
    def __init__(self, dataframe, tokenizer, bert_model, max_seq_len=128):
        self.texts = dataframe.text.values.tolist()
        self.labels = dataframe.stance.values.tolist() if 'stance' in dataframe else None
        self.max_seq_len = max_seq_len  # Max sequence length for BERT
        self.tokenizer = tokenizer  # ✅ Store tokenizer
        self.bert_model = bert_model 

    def _get_bert_embeddings(self, sentences):
        """
        Tokenizes input text and returns 128-dimension BERT embeddings.
        """
        encoded_input = tokenizer(sentences, padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors="pt")

        with torch.no_grad():
            model_output = bert_model(**encoded_input)  # Extract BERT embeddings
        #print(f"Shape of BERT output: {model_output.last_hidden_state.shape}")
        return model_output.last_hidden_state.mean(dim=1)  # [batch_size, 128]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_embedding = self._get_bert_embeddings([self.texts[idx]])[0]  # [128]

        if self.labels is not None:
            label = self.labels[idx]
            return text_embedding, torch.tensor(label, dtype=torch.long)
        else:
            return text_embedding
def create_rnn_loaders_bert(df, batch_size, tokenizer, bert_model, shuffle):
    dataset = TweetDatasetBertEmbeddings(df, tokenizer, bert_model)  # No more target word
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader