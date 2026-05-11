# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BiCond(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(BiCond, self).__init__()

#         self.hidden = 265
#         self.dropout = nn.Dropout(0.1)
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         self.target_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)
#         self.text_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)
#         self.classifier = nn.Linear(self.hidden * 2, num_classes)

#     def forward(self, x, x_len, epoch, target_word=None):
#         document_enc = self.embedding(x)  # batch maxlen glove-hidden
#         target_enc = self.embedding(target_word)

#         batch_size = document_enc.shape[0]
#         max_length = document_enc.shape[1]
#         hidden_dim = document_enc.shape[2]

#         # target encoding
#         _, target_last_hn_cn = self.target_bilstm(target_enc)
#         # document encoding
#         _, (txt_last_hn, c) = self.text_bilstm(document_enc, target_last_hn_cn)
#         txt_last_hn = self.dropout(txt_last_hn)

#         output = txt_last_hn.transpose(0, 1).reshape((-1, 2 * self.hidden))  # the last output of text_bilstm
#         output = self.classifier(output)
#         # output = F.tanh(output)
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class BiCond(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super(BiCond, self).__init__()

        self.hidden = 512
        self.dropout = nn.Dropout(0.2)
        

        # Load pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
        
        # BiLSTM for Target
        self.target_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)
        
        # BiLSTM for Document (which will be initialized with target encoding)
        self.text_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True, dropout=0.2)
        
        # Final classifier
        self.classifier = nn.Linear(self.hidden * 2, num_classes)

    def forward(self, x, x_len, epoch, target_word=None):
        """
        Forward pass for BiCond Model
        
        Parameters:
        - x: Input document (word indices)
        - x_len: Document lengths
        - epoch: Current training epoch (not used in forward pass)
        - target_word: Input target word sequence (word indices)

        Returns:
        - output: Class probabilities (log softmax)
        """
        # Get embeddings
        document_enc = self.embedding(x)  # Shape: (batch, max_len, embedding_dim)
        target_enc = self.embedding(target_word)  # Shape: (batch, target_len, embedding_dim)

        # Encode the target (bidirectional)
        _, (target_hn, target_cn) = self.target_bilstm(target_enc)  # Expected shape: (2, batch, hidden)

        # Directly use target_hn and target_cn as they are in correct shape
        text_init_hn = target_hn  # (2, batch, hidden)
        text_init_cn = target_cn  # (2, batch, hidden)

        # Encode the document, initializing with conditioned target state
        _, (txt_last_hn, txt_last_cn) = self.text_bilstm(document_enc, (text_init_hn, text_init_cn))

        # Concatenate forward and backward hidden states from document encoding
        txt_last_hn = torch.cat((txt_last_hn[0], txt_last_hn[1]), dim=1)  # Shape: (batch, 2*hidden)

        # Apply dropout and classification
        output = self.dropout(txt_last_hn)
        output = self.classifier(output)
        #output = F.log_softmax(output, dim=1)  # Log probabilities for classification

        return output