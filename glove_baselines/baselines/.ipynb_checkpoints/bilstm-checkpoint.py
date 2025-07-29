# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BiLSTMModel(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(BiLSTMModel, self).__init__()

#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         self.dropout1 = nn.Dropout(0.1)
#         self.lstm1 = nn.LSTM(embedding_matrix.shape[1], 265, batch_first=True, dropout=0.2)
#         self.bilstm = nn.LSTM(265, 265, batch_first=True, dropout=0.2, bidirectional=True)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc = nn.Linear(265*2, num_classes)  # Times 2 because of the bidirectional LSTM

#     def forward(self, x, x_len, epoch, target_word=None):
#         x = self.embedding(x)
#         x = self.dropout1(x)
#         x, _ = self.lstm1(x)
#         x, _ = self.bilstm(x)
#         x = x[:, -1, :]  # Taking the last output for classification
#         x = self.dropout2(x)
#         x = self.fc(x)
#         return x



import torch
import torch.nn as nn
import torch.nn.functional as F

# class BiLSTMModel(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(BiLSTMModel, self).__init__()

#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         self.dropout1 = nn.Dropout(0.1)

#         # First LSTM (Unidirectional)
#         self.lstm1 = nn.LSTM(embedding_matrix.shape[1], 256, batch_first=True)

#         # Second BiLSTM (Bidirectional)
#         self.bilstm = nn.LSTM(256, 512, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)

#         # Dropout before classification
#         self.dropout2 = nn.Dropout(0.2)

#         # Fully connected layer
#         self.fc = nn.Linear(512 * 2, num_classes)  # Corrected hidden size for BiLSTM

#     def forward(self, x, x_len, epoch, target_word=None):
#         x = self.embedding(x)
#         x = self.dropout1(x)

#         x, _ = self.lstm1(x)  # First LSTM layer
#         x = self.dropout1(x)  # Apply dropout manually

#         x, _ = self.bilstm(x)  # BiLSTM layer

#         # Mean pooling over time steps instead of taking only last hidden state
#         x = torch.mean(x, dim=1)  

#         x = self.dropout2(x)  # Dropout before classification
#         x = self.fc(x)
#         return x  # No softmax (CrossEntropyLoss expects raw logits)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super(BiLSTMModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
        self.dropout1 = nn.Dropout(0.1)

        # First LSTM (Unidirectional)
        self.lstm1 = nn.LSTM(embedding_matrix.shape[1], 256, batch_first=True)

        # Second BiLSTM (Bidirectional)
        self.bilstm = nn.LSTM(256, 512, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)  # ✅ Fixed input size

        # Dropout before classification
        self.dropout2 = nn.Dropout(0.2)

        # Fully connected layer
        self.fc = nn.Linear(512 * 2, num_classes)  # Hidden size * 2 for BiLSTM

    def forward(self, x, x_len, epoch, target_word=None):
        # Embedding Layer
        x = self.embedding(x)
        x = self.dropout1(x)

        # **Sort input by length for PackedSequence processing**
        seq_lengths = torch.tensor(x_len, dtype=torch.long, device=x.device)  # ✅ Ensure x_len is a tensor
        seq_lengths, perm_idx = seq_lengths.sort(descending=True)
        x = x[perm_idx]  # Reorder input based on length

        # **Pack sequences for LSTM**
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # First LSTM (Unidirectional)
        packed_output, _ = self.lstm1(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # **Restore original order**
        _, unperm_idx = perm_idx.sort()
        output = output[unperm_idx]

        # BiLSTM Layer
        output, _ = self.bilstm(output)

        # Mean pooling over time steps
        output = torch.mean(output, dim=1)

        # Dropout & Classification
        output = self.dropout2(output)
        output = self.fc(output)

        return output  # Raw logits (CrossEntropyLoss applies softmax)