

# from torch import nn

# # class RobertaModel(nn.Module):
# #     def __init__(self, base_model, num_classes):
# #         super(RobertaModel, self).__init__()

# #         self.bert = base_model
# #         self.fc1 = nn.Linear(768, 32)
# #         # Dynamically set the output size based on the number of classes
# #         self.fc2 = nn.Linear(32, num_classes)

# #         self.relu = nn.ReLU()
# #         # Softmax is not applied here if using nn.CrossEntropyLoss
        
# #     def forward(self, input_ids, attention_mask):
# #         bert_out = self.bert(input_ids=input_ids,
# #                              attention_mask=attention_mask)[0][:, 0]
# #         x = self.fc1(bert_out)
# #         x = self.relu(x)
        
# #         x = self.fc2(x)
# #         # No softmax applied in forward if using nn.CrossEntropyLoss
# #         return x


import torch
from torch import nn
class BERTModel(nn.Module):
    def __init__(self, base_model, num_classes):
        """
        Generic model for BERT/RoBERTa-style transformers.

        Args:
            base_model (AutoModel): Pretrained transformer model (BERT, RoBERTa, etc.).
            num_classes (int): Number of output classes.
        """
        super(BERTModel, self).__init__()
        self.num_classes = num_classes
        self.transformer = base_model  # Load BERT/RoBERTa
        self.fc1 = nn.Linear(768, 256)  # Hidden layer
        self.fc2 = nn.Linear(256, 1 if num_classes == 2 else self.num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (Tensor): Tokenized input sequences.
            attention_mask (Tensor): Attention masks for padding.

        Returns:
            Tensor: Raw logits (no softmax, as nn.CrossEntropyLoss applies it).
        """
        transformer_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract CLS token representation (batch_size, hidden_dim)
        cls_embedding = transformer_out.last_hidden_state[:, 0]  

        x = self.fc1(cls_embedding)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Final output logits
        if self.num_classes == 2:
            return x.squeeze(1)
        return x  # Logits (used with nn.CrossEntropyLoss)



# class BERTModel(nn.Module):
#     def __init__(self, base_model, num_classes=3):
#         """
#         Dual Attention Network with BERT.

#         Args:
#             base_model (AutoModel): Pretrained BERT model.
#             num_classes (int): Number of stance classes (default=3).
#         """
#         super(BERTModel, self).__init__()

#         self.hidden_dim = 256   # Fixed hidden dimension
#         self.dropout = 0.2      # Fixed dropout rate

#         # Objective & Subjective Views (Two BERTs)
#         self.F_obj = base_model  # First BERT for objective features
#         self.F_subj = base_model  # Second BERT for subjective features

#         # Objective View Classifiers
#         self.objective_domain_discriminator = nn.Sequential(
#             nn.Linear(768, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_dim, 1)
#         )

#         self.objective_classifier = nn.Sequential(
#             nn.Linear(768, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_dim, 2)  # Binary classification
#         )

#         # Subjective View Classifiers
#         self.subjective_domain_discriminator = nn.Sequential(
#             nn.Linear(768, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_dim, 1)
#         )

#         self.subjective_classifier = nn.Sequential(
#             nn.Linear(768, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_dim, 2)  # Binary classification
#         )

#         # Fusion Layer
#         self.g = nn.Sequential(
#             nn.Linear(1536, 768),
#             nn.Sigmoid()
#         )

#         # Stance Classification Layer
#         self.stance_classifier = nn.Sequential(
#             nn.Linear(768, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.hidden_dim, num_classes)  # Stance classification
#         )

#     def forward(self, input_ids, attention_mask, a=None):
#         """
#         Forward pass for BERT-based stance detection.

#         Args:
#             input_ids (Tensor): Tokenized input sequences.
#             attention_mask (Tensor): Attention masks for padding.
#             a (Optional): If provided, returns additional auxiliary outputs.

#         Returns:
#             - Stance prediction (batch_size, num_classes).
#             - (If a is not None) Additional outputs for auxiliary tasks.
#         """
#         f_obj = self.F_obj(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]  # CLS Token
#         f_subj = self.F_subj(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]  # CLS Token

#         # Fusion
#         f_obj_subj = torch.cat((f_obj, f_subj), dim=1)  # (batch_size, 1536)
#         g = self.g(f_obj_subj)
#         f_dual = (g * f_subj) + ((1 - g) * f_obj)
#         stance_prediction = self.stance_classifier(f_dual)

#         if a is not None:
#             objective_prediction = self.objective_classifier(f_obj)
#             subjective_prediction = self.subjective_classifier(f_subj)

#             objective_domain_prediction = self.objective_domain_discriminator(f_obj)
#             subjective_domain_prediction = self.subjective_domain_discriminator(f_subj)

#             return stance_prediction, objective_prediction, subjective_prediction, objective_domain_prediction, subjective_domain_prediction

#         return stance_prediction



# class BERTModel(nn.Module):
#     def __init__(self, base_model, num_classes):
#         """
#         Extended BERT model with additional layers.

#         Args:
#             base_model (AutoModel): Pretrained transformer model (BERT, RoBERTa, etc.).
#             num_classes (int): Number of output classes.
#         """
#         super(BERTModel, self).__init__()

#         self.transformer = base_model  # Load BERT/RoBERTa

#         # Adding more hidden layers
#         self.fc1 = nn.Linear(768, 256)  # Expanded hidden layer
#         self.fc2 = nn.Linear(256, 128)  # Additional hidden layer
#         self.fc3 = nn.Linear(128, 64)   # Another hidden layer
#         self.fc4 = nn.Linear(64, num_classes)  # Final classification layer

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)  # Dropout to prevent overfitting

#     def forward(self, input_ids, attention_mask):
#         """
#         Forward pass.

#         Args:
#             input_ids (Tensor): Tokenized input sequences.
#             attention_mask (Tensor): Attention masks for padding.

#         Returns:
#             Tensor: Raw logits (no softmax, as nn.CrossEntropyLoss applies it).
#         """
#         transformer_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

#         # Extract CLS token representation (batch_size, hidden_dim)
#         cls_embedding = transformer_out.last_hidden_state[:, 0]  

#         # Passing through additional layers
#         x = self.fc1(cls_embedding)
#         x = self.relu(x)
#         #x = self.dropout(x)

#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.fc4(x)  # Final output logits

#         return x  # Logits (used with nn.CrossEntropyLoss)