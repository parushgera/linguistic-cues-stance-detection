import torch
import torch.nn as nn
import torch.nn.functional as F


# class BiGRUModel(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(BiGRUModel, self).__init__()

#         # Embedding layer with pretrained weights
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         self.dropout1 = nn.Dropout(0.1)

#         # First GRU (Unidirectional)
#         self.gru1 = nn.GRU(embedding_matrix.shape[1], 256, batch_first=True)

#         # Second BiGRU (Bidirectional)
#         self.bigru = nn.GRU(256, 256, batch_first=True, num_layers=1, dropout=0.2, bidirectional=True)

#         # Dropout before classification
#         self.dropout2 = nn.Dropout(0.3)

#         # Fully connected layer
#         self.fc = nn.Linear(256 * 2, num_classes)  # Hidden size * 2 for BiGRU

#     def forward(self, x, x_len, epoch, target_word=None):
#         x = self.embedding(x)
#         x = self.dropout1(x)

#         x, _ = self.gru1(x)  # First GRU layer
#         x = self.dropout1(x)  # Apply dropout manually

#         x, _ = self.bigru(x)  # BiGRU layer

#         # Mean pooling over time steps instead of taking only last hidden state
#         x = torch.mean(x, dim=1)  

#         x = self.dropout2(x)  # Dropout before classification
#         x = self.fc(x)
#         return x  # No softmax (CrossEntropyLoss expects raw logits)
    
    
    
# class ATBiGRU(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(ATBiGRU, self).__init__()

#         self.hidden_size = 512  # Size of GRU hidden states
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)

#         # BiGRU Layer
#         self.bigru = nn.GRU(embedding_matrix.shape[1], self.hidden_size, batch_first=True, bidirectional=True)

#         # Attention Mechanism
#         self.W = nn.Linear(2 * self.hidden_size, self.hidden_size)  # Project to hidden size
#         self.u = nn.Linear(self.hidden_size, 1)  # Final attention scores
#         self.tanh = nn.Tanh()

#         # Fully connected layer
#         self.fc = nn.Linear(2 * self.hidden_size, num_classes)

#     def attention(self, rnn_output):
#         """
#         Attention mechanism applied to BiGRU output.
#         """
#         # Project BiGRU output to attention space
#         attn_weights = self.tanh(self.W(rnn_output))  # (batch, seq_len, hidden_size)

#         # Compute attention scores
#         attn_weights = self.u(attn_weights)  # (batch, seq_len, 1)
#         attn_weights = F.softmax(attn_weights, dim=1)  # Normalize attention scores

#         # Compute weighted sum of BiGRU outputs
#         context_vector = torch.sum(rnn_output * attn_weights, dim=1)  # (batch, 2*hidden_size)

#         return context_vector

#     def forward(self, x, x_len, epoch, target_word=None):
#         x = self.embedding(x)  # Embedding layer

#         rnn_output, _ = self.bigru(x)  # BiGRU Layer

#         context_vector = self.attention(rnn_output)  # Apply attention

#         output = self.fc(context_vector)  # Fully connected layer

#         return output  # Raw logits (use CrossEntropyLoss)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ATBiGRU(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout, num_layers):
        super(ATBiGRU, self).__init__()

        self.hidden_size = hidden_size  # Size of GRU hidden states
        self.input_size = 384  # BERT embedding dimension
        self.dropout = dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_layers = num_layers
        # BiGRU Layer
        self.bigru = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        # Attention Mechanism
        self.W = nn.Linear(2 * self.hidden_size, self.hidden_size)  # Project to hidden size
        self.u = nn.Linear(self.hidden_size, 1)  # Final attention scores
        self.tanh = nn.Tanh()

        # Fully connected layer
        #self.fc = nn.Linear(2 * self.hidden_size, self.num_classes)
        self.fc = nn.Linear(2 * self.hidden_size, 1 if num_classes == 2 else num_classes)

    def attention(self, rnn_output):
        """
        Attention mechanism applied to BiGRU output.
        """
        # Project BiGRU output to attention space
        attn_weights = self.tanh(self.W(rnn_output))  # (batch, seq_len, hidden_size)

        # Compute attention scores
        attn_weights = self.u(attn_weights)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize attention scores

        # Compute weighted sum of BiGRU outputs
        context_vector = torch.sum(rnn_output * attn_weights, dim=1)  # (batch, 2*hidden_size)
        

        return context_vector

    def forward(self, text_emb):
        """
        text_emb: (batch_size, 384) -> Precomputed BERT embeddings
        """

        # Reshape input to match BiGRU expected shape (batch, seq_len=1, embedding_dim=384)
        #x = text_emb.unsqueeze(1)  # Shape: (batch, 1, 384)
        x = text_emb
        x = self.dropout1(x)
        # BiGRU layer
        rnn_output, _ = self.bigru(x)  # (batch, 1, 2*hidden_size)

        # Apply attention
        context_vector = self.attention(rnn_output)  # (batch, 2*hidden_size)
        context_vector = self.dropout2(context_vector)
        

        # Fully connected layer
        output = self.fc(context_vector)  # (batch, num_classes)
        if self.num_classes == 2:
            output = output.squeeze(1)

        return output  # Return raw logits (use CrossEntropyLoss)