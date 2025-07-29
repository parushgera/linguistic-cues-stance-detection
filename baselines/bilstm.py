import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



# class BiLSTM(nn.Module):
#     def __init__(self, num_classes, hidden_size, dropout, num_layers):
#         super(BiLSTM, self).__init__()

#         self.hidden_size = hidden_size  # Hidden size of LSTM
#         self.input_size = 384  # BERT embedding dimension
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.num_classes = num_classes

#         # First LSTM (Unidirectional)
#         self.lstm1 = nn.LSTM(self.input_size, 256, batch_first=True)

#         # Second BiLSTM (Bidirectional)
#         self.bilstm = nn.LSTM(256, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=True)


#         # Dropout before classification
#         self.dropout1 = nn.Dropout(self.dropout)
#         self.dropout2 = nn.Dropout(self.dropout)

#         # Fully connected layer
#         #self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)  # Hidden size * 2 for BiLSTM
#         self.fc = nn.Linear(2 * self.hidden_size, 1 if num_classes == 2 else num_classes)

#     def forward(self, text_emb):
#         """
#         text_emb: (batch_size, 384) -> Precomputed BERT embeddings
#         """

#         # Reshape for BiLSTM input (batch, seq_len=1, embedding_dim=384)
#         x = text_emb.unsqueeze(1)  # Shape: (batch, 1, 384)
#         x = self.dropout1(x)

#         # First LSTM (Unidirectional)
#         output, _ = self.lstm1(x)

#         # BiLSTM Layer
#         output, _ = self.bilstm(output)

#         # Mean pooling over time steps
#         output = torch.mean(output, dim=1)

#         # Dropout & Classification
#         output = self.dropout2(output)
#         output = self.fc(output)
#         if self.num_classes == 2:
#             output = output.squeeze(1)
#         return output  # Raw logits (CrossEntropyLoss applies softmax)


class BiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout, num_layers):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size  # LSTM hidden size
        self.input_size = 384  # BERT token embedding size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=self.input_size,  # 384
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # Ensures (batch_size, seq_len, feature_dim)
            dropout=self.dropout,
            bidirectional=True
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

        # Fully connected output layer
        self.fc = nn.Linear(2 * self.hidden_size, 1 if num_classes == 2 else num_classes)

    def forward(self, text_emb):
        """
        text_emb: (batch_size, seq_len=128, embedding_dim=384) -> Token-level BERT embeddings
        """

        # ✅ Debugging step: Ensure input shape
        #print(f"🔍 Input Shape Before LSTM: {text_emb.shape}")  # Expected: (batch_size, 128, 384)

        # Ensure correct shape before passing into LSTM
        if text_emb.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, embedding_dim), got {text_emb.shape}")

        # Apply dropout before BiLSTM
        #print(text_emb.shape)
        x = text_emb
        x = self.dropout1(x)  # Shape: (batch_size, 128, 384)

        # BiLSTM Layer
        output, (h_n, c_n) = self.bilstm(x)  # output: (batch_size, seq_len=128, hidden_size * 2)

        # ✅ Debugging step: Ensure output shape
        #print(f"🔍 LSTM Output Shape: {output.shape}")  # Expected: (batch_size, seq_len=128, hidden_size * 2)

        # Mean Pooling over all time steps
        output = torch.mean(output, dim=1)  # (batch_size, hidden_size * 2)

        # Apply dropout before classification
        output = self.dropout2(output)
        output = self.fc(output)

        if self.num_classes == 2:
            output = output.squeeze(1)

        return output