# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class TextCNN(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super().__init__()
#         self.hidden = 265

#         # Adjustments: Assuming you want to apply 1D convolutions across the sequence (word embeddings)
#         output_channel = 100
#         kernel_sizes = [3, 4, 5]

#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         # Conv1d: in_channels (embedding dim), out_channels, kernel_size
#         self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=output_channel, kernel_size=k) for k in kernel_sizes])
#         self.q = nn.Linear(len(kernel_sizes) * output_channel, len(kernel_sizes) * output_channel)
#         self.k = nn.Linear(len(kernel_sizes) * output_channel, len(kernel_sizes) * output_channel)
#         self.v = nn.Linear(len(kernel_sizes) * output_channel, len(kernel_sizes) * output_channel)
#         self.classifier = nn.Linear(len(kernel_sizes) * output_channel, num_classes)

#     def forward(self, x, x_len, epoch, target_word=None):
#         x = self.embedding(x)  # [batch, maxlen, embedding_dim]
#         x = x.transpose(1, 2)  # [batch, embedding_dim, maxlen] to fit Conv1d input requirements

#         x = [F.relu(conv(x)).squeeze(2) for conv in self.convs]  # Apply Conv1d
#         x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Pooling
#         x = torch.cat(x, dim=1)  # Concatenate the feature maps

#         # self-attention
#         query = self.q(x)
#         key = self.k(x)
#         attn = torch.mm(query, key.transpose(0, 1))
#         x = torch.mm(attn, self.v(x))

#         output = self.classifier(x)

#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F

# class TextCNN(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(TextCNN, self).__init__()
#         output_channel = 100  # Number of filters per kernel size
#         kernel_sizes = [3, 4, 5]  # Different n-gram sizes

#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float().clone().detach(), freeze=True)

#         # Convolutional layers
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=output_channel, kernel_size=k)
#             for k in kernel_sizes
#         ])

#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.5)

#         # Self-attention mechanism
#         feature_dim = len(kernel_sizes) * output_channel
#         self.q = nn.Linear(feature_dim, feature_dim)
#         self.k = nn.Linear(feature_dim, feature_dim)
#         self.v = nn.Linear(feature_dim, feature_dim)

#         # Fully connected classifier
#         self.classifier = nn.Linear(feature_dim, num_classes)

#     def forward(self, x, x_len, epoch, target_word=None):
#         x = self.embedding(x)  # [batch, max_len, embedding_dim]
#         x = x.transpose(1, 2)  # [batch, embedding_dim, max_len] for Conv1D

#         # Apply Conv1D + ReLU activation
#         x = [F.relu(conv(x)).squeeze(2) for conv in self.convs]  

#         # Apply max pooling over time dimension
#         x = [F.adaptive_max_pool1d(i, 1).squeeze(2) for i in x]  
#         x = torch.cat(x, dim=1)  # Concatenate different filter outputs

#         # Self-attention mechanism (corrected)
#         query = self.q(x).unsqueeze(1)  # Shape: [batch_size, 1, feature_dim]
#         key = self.k(x).unsqueeze(2)  # Shape: [batch_size, feature_dim, 1]

#         # Compute attention scores using batch matrix multiplication
#         attn = torch.bmm(query, key)  # Shape: [batch_size, 1, 1]
#         attn = F.softmax(attn, dim=-1)  # Normalize attention weights

#         # Apply attention to value (V)
#         x = self.v(x) * attn.squeeze(2)  # Element-wise multiplication with correct shape

#         # Apply dropout before classification
#         x = self.dropout(x)

#         # Classifier
#         output = self.classifier(x)  # No softmax applied

#         return output  # Return raw logits (CrossEntropyLoss will apply softmax)

class KimCNN(nn.Module): #best working
    def __init__(self, embedding_matrix, num_classes):
        super(KimCNN, self).__init__()
        output_channel = 25  # Number of filters per kernel size (from your paper)
        kernel_sizes = [2, 3, 4, 5]  # N-gram filter sizes

        # Embedding Layer
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float().clone().detach(), freeze=True)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=output_channel, kernel_size=k)
            for k in kernel_sizes
        ])

        # Fully connected classifier (hidden dimension = 128, per your paper)
        self.fc1 = nn.Linear(len(kernel_sizes) * output_channel, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, x_len, epoch, target_word=None):
        x = self.embedding(x)  # [batch, max_len, embedding_dim]
        x = x.transpose(1, 2)  # Reshape for Conv1D: [batch, embedding_dim, max_len]

        # Apply Conv1D + ReLU activation
        x = [F.relu(conv(x)).squeeze(2) for conv in self.convs]  

        # Apply max pooling over time dimension
        x = [F.adaptive_max_pool1d(i, 1).squeeze(2) for i in x]  
        x = torch.cat(x, dim=1)  # Concatenate different filter outputs

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))  # First FC layer with ReLU
        x = self.fc2(x)  # Output layer

        return x  # Return raw logits (CrossEntropyLoss applies softmax)