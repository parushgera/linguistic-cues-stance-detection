


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, num_classes, dropout, fc1_size):
        super(KimCNN, self).__init__()
        output_channel = 25  # Number of filters per kernel size
        kernel_sizes = [2, 3, 4, 5]  # N-gram filter sizes
        embedding_dim = 384  # MiniLM-BERT output
        self.dropout = dropout
        self.fc1_size = fc1_size
        self.num_classes = num_classes

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=output_channel, kernel_size=k, padding='valid')
            for k in kernel_sizes
        ])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(output_channel) for _ in kernel_sizes])

        # Fully connected classifier
        self.fc1 = nn.Linear(len(kernel_sizes) * output_channel, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, 1 if num_classes == 2 else num_classes)

        # Dropout
        self.dropout1 = nn.Dropout(self.dropout)

    def forward(self, text_emb):
        """
        text_emb: (batch_size, seq_len, embedding_dim)
        Expected shape: (batch_size, 128, 384) -> Needs (batch_size, 384, 128) for Conv1D
        """

        # Permute input to match Conv1D format: (batch_size, embedding_dim, seq_len)
        x = text_emb.permute(0, 2, 1)  # Shape: (batch_size, 384, 128)

        # Apply Conv1D + BatchNorm + ReLU
        x = [F.relu(bn(conv(x))) for conv, bn in zip(self.convs, self.batch_norms)]

        # Apply Max Pooling
        x = [F.adaptive_max_pool1d(i, 1).squeeze(2) for i in x]
        x = torch.cat(x, dim=1)  # Concatenate feature maps

        # Fully Connected Layers
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)

        if self.num_classes == 2:
            return x.squeeze(1)  # For BCEWithLogitsLoss

        return x  # For CrossEntropyLoss


# class KimCNN(nn.Module):
#     def __init__(self, num_classes, dropout, fc1_size):
#         super(KimCNN, self).__init__()
#         output_channel = 25  # Number of filters per kernel size
#         kernel_sizes = [2, 3, 4, 5]  # ✅ Ensure sequence length >= 5
#         embedding_dim = 384  # MiniLM-BERT output
#         self.dropout = dropout
#         self.fc1_size = fc1_size 
#         self.num_classes = num_classes

#         # Convolutional layers
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=embedding_dim, out_channels=output_channel, kernel_size=k, padding='valid')
#             for k in kernel_sizes
#         ])
#         self.batch_norms = nn.ModuleList([nn.BatchNorm1d(output_channel) for _ in kernel_sizes])

#         # Fully connected classifier
#         self.fc1 = nn.Linear(len(kernel_sizes) * output_channel, self.fc1_size)
#         self.fc2 = nn.Linear(self.fc1_size, 1 if num_classes == 2 else num_classes)

#         # Dropout
#         self.dropout1 = nn.Dropout(self.dropout)

#     def forward(self, text_emb):
#         """
#         text_emb: (batch_size, seq_len, embedding_dim)
#         Expected shape: (batch_size, 128, 384) -> Needs (batch_size, 384, 128) for Conv1D
#         """

#         #print('Here is shape', text_emb.shape)
#         x = text_emb  # Now shape: (batch_size, 384, 128)

#         # Apply Conv1D + BatchNorm + ReLU
#         x = [F.relu(bn(conv(x))) for conv, bn in zip(self.convs, self.batch_norms)]

#         # Apply Max Pooling
#         x = [F.adaptive_max_pool1d(i, 1).squeeze(2) for i in x]
#         x = torch.cat(x, dim=1)  # Concatenate feature maps

#         # Fully Connected Layers
#         x = self.dropout1(F.relu(self.fc1(x)))
#         x = self.fc2(x)

#         if self.num_classes == 2:
#             return x.squeeze(1)  # For BCEWithLogitsLoss

#         return x  # For CrossEntropyLoss

# class KimCNN(nn.Module):
#     def __init__(self, num_classes, dropout, fc1_size):
#         super(KimCNN, self).__init__()
#         output_channel = 25  # Number of filters per kernel size
#         kernel_sizes = [1]  # Since BERT embeddings are single-dimension vectors, use kernel size 1
#         embedding_dim = 384  # MiniLM-BERT output
#         self.dropout = dropout
#         self.fc1_size = fc1_size 
#         self.num_classes = num_classes

#         # Convolutional layers with correct in_channels
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=embedding_dim, out_channels=output_channel, kernel_size=k)
#             for k in kernel_sizes
#         ])
#         self.batch_norms = nn.ModuleList([
#             nn.BatchNorm1d(output_channel) for _ in kernel_sizes
#         ])

#         # Fully connected classifier
#         self.fc1 = nn.Linear(len(kernel_sizes) * output_channel, self.fc1_size)
#         self.fc2 = nn.Linear(self.fc1_size, 1 if num_classes == 2 else num_classes)

#         # Dropout
#         self.dropout1 = nn.Dropout(self.dropout)

#     def forward(self, text_emb):
#         """
#         text_emb: (batch_size, 384) -> Text BERT embeddings
#         """

#         # Reshape for Conv1D: (batch, embedding_dim, sequence_length=1)
#         x = text_emb.unsqueeze(2)  # Shape: (batch_size, embedding_dim, 1)

#         # Apply Conv1D + BatchNorm + ReLU activation
#         x = [F.relu(bn(conv(x))).squeeze(2) for conv, bn in zip(self.convs, self.batch_norms)]

#         # Concatenate convolution outputs
#         x = torch.cat(x, dim=1)  # Shape: (batch_size, total_filters)

#         # Fully connected layers
#         x = self.dropout1(F.relu(self.fc1(x)))  # First FC layer with ReLU
#         x = self.fc2(x)  # Output layer
        
#         # Adjust output shape for binary classification
#         if self.num_classes == 2:
#             return x.squeeze(1)  # BCEWithLogitsLoss expects shape (batch_size,)

#         return x  # Return raw logits (CrossEntropyLoss applies softmax)
    
    
    
    
    
# class KimCNN(nn.Module):
#     def __init__(self, num_classes, dropout, fc1_size):
#         super(KimCNN, self).__init__()
#         output_channel = 25  # Number of filters per kernel size
#         kernel_sizes = [2, 3, 4, 5]  # N-gram filter sizes
#         embedding_dim = 384  # MiniLM-BERT output
#         self.dropout = dropout
#         self.fc1_size = fc1_size 
#         self.num_classes = num_classes

#         # Convolutional layers
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=1, out_channels=output_channel, kernel_size=k)
#             for k in kernel_sizes
#         ])

#         # Fully connected classifier
#         self.fc1 = nn.Linear(len(kernel_sizes) * output_channel, self.fc1_size)
#         self.fc2 = nn.Linear(self.fc1_size, 1 if num_classes == 2 else num_classes)

#         # Dropout
#         self.dropout1 = nn.Dropout(self.dropout)

#     def forward(self, text_emb):
#         """
#         text_emb: (batch_size, 384) -> Text BERT embeddings
#         """

#         #print(f"Before Reshape: {text_emb.shape}")  # Expected: (batch_size, 384)

#         # Reshape for Conv1D (batch, channels=1, width=384)
#         x = text_emb.unsqueeze(1)  # Shape: (batch, 1, 384)

#         #print(f"After Reshape: {x.shape}")  # Expected: (batch_size, 1, 384)

#         # Apply Conv1D + ReLU activation
#         x = [F.relu(conv(x)).squeeze(2) for conv in self.convs]

#         # Apply max pooling over time dimension
#         x = [F.adaptive_max_pool1d(i, 1).squeeze(2) for i in x]  
#         x = torch.cat(x, dim=1)  # Concatenate different filter outputs

#         # Fully connected layers
#         x = self.dropout1(F.relu(self.fc1(x)))  # First FC layer with ReLU
#         x = self.fc2(x)  # Output layer
#         if self.num_classes == 2:
#             return x.squeeze(1)

#         return x  # Return raw logits (CrossEntropyLoss applies softmax)