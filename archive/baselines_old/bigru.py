import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout, num_layers):
        super(BiGRU, self).__init__()

        self.hidden_size = hidden_size  # GRU hidden size
        self.input_size = 384  # BERT embedding dimension
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # BiGRU Layer
        self.bigru = nn.GRU(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True
        )

        # Dropout layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected layer for classification
        #self.fc = nn.Linear(2 * self.hidden_size, num_classes)  # Bidirectional -> hidden_size * 2
        self.fc = nn.Linear(2 * self.hidden_size, 1 if num_classes == 2 else num_classes)

    def forward(self, text_emb):
        """
        text_emb: (batch_size, 384) -> Precomputed BERT embeddings
        """

        # Reshape input for BiGRU (batch, seq_len=1, embedding_dim=384)
        #x = text_emb.unsqueeze(1)  # Shape: (batch, 1, 384)
        x = text_emb
        x = self.dropout1(x)

        # BiGRU layer
        rnn_output, _ = self.bigru(x)  # (batch, seq_len=1, 2*hidden_size)

        # Take the final hidden state (forward + backward)
        output = rnn_output[:, -1, :]  # Shape: (batch, 2*hidden_size)

        # Dropout before classification
        output = self.dropout2(output)

        # Fully connected classification layer
        output = self.fc(output)  # Shape: (batch, num_classes)
        if self.num_classes == 2:
            output = output.squeeze(1)

        return output  # Raw logits (CrossEntropyLoss applies softmax)
