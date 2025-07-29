# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CrossNet(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(CrossNet, self).__init__()
#         self.hidden = 265
#         self.dropout = nn.Dropout(0.1)
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         self.target_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)
#         self.text_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)
#         self.W = nn.Linear(2 * self.hidden, self.hidden)
#         self.w = nn.Linear(self.hidden, 1)
#         self.act = nn.Tanh()
#         self.classifier = nn.Linear(self.hidden * 2, num_classes)
#         self.softmax = nn.Softmax()

#     def forward(self, x, x_len, epoch, target_word=None):
#         # embedding layer
#         document_enc = self.embedding(x)  # batch maxlen glove-hidden
#         target_enc = self.embedding(target_word)

#         # Context encoding layer
#         _, target_last_hn_cn = self.target_bilstm(target_enc)  # target encoding
#         last_hidden_output, _ = self.text_bilstm(document_enc, target_last_hn_cn)  # document encoding
#         last_hidden_output = self.dropout(last_hidden_output)
#         # print(last_hidden_output.shape) [batchsize, seqlen(maxlen), 2*hidden]

#         # Aspect Attention Layer
#         coeff = self.W(last_hidden_output)
#         coeff = self.act(coeff)
#         coeff = self.w(coeff)  # [batch, seqlen, 1]

#         attn = F.softmax(coeff, dim=-1)  # [batch, seqlen, 1]
#         output = torch.mul(last_hidden_output, attn)  # the second one could be broadcast

#         '''
#         # same as above
#         attn_exp = attn.expand(attn.shape[0], attn.shape[1], 2*self.hidden)
#         outputs = torch.mul(last_hidden_output, attn_exp)
#         '''

#         # Prediction Layer
#         output = torch.sum(output, dim=1)  # [batchsize, 2*hidden]
#         output = self.classifier(output)
#         # output = self.softmax(output)

#         return output



import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossNet(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(CrossNet, self).__init__()
#         self.hidden = 265
#         self.dropout = nn.Dropout(0.1)
        

#         # Pre-trained embeddings
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)

#         # BiLSTM for Target Encoding
#         self.target_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)

#         # BiLSTM for Document Encoding (Conditioned on Target)
#         self.text_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, num_layers=1, dropout=0.2, bidirectional=True, batch_first=True)

#         # Attention Mechanism
#         self.W = nn.Linear(2 * self.hidden, self.hidden)
#         self.w = nn.Linear(self.hidden, 1)
#         self.act = nn.Tanh()

#         # Final Classification
#         self.classifier = nn.Linear(self.hidden * 2, num_classes)

#     def forward(self, x, x_len, epoch, target_word=None):
#         # Embedding
#         document_enc = self.embedding(x) # Shape: (batch, max_len, embedding_dim)
#         target_enc = self.embedding(target_word)  # Shape: (batch, target_len, embedding_dim)

#         # Target BiLSTM Encoding
#         _, (target_hn, target_cn) = self.target_bilstm(target_enc)

#         # Document BiLSTM Encoding (Conditioned on Target)
#         last_hidden_output, _ = self.text_bilstm(document_enc, (target_hn, target_cn))
#         last_hidden_output = self.dropout(last_hidden_output)  # Regularization

#         # Attention Mechanism
#         coeff = self.W(last_hidden_output)
#         coeff = self.act(coeff)
#         coeff = self.w(coeff)  
#         attn = F.softmax(coeff, dim=1)  
#         attn = self.dropout(attn)  # Regularization
        
#         output = torch.mul(last_hidden_output, attn)
#         output = torch.sum(output, dim=1)  # Weighted sum over words

#         # Classification
#         output = self.classifier(output)

#         return output  # Raw logits (use CrossEntropyLoss)



class CrossNet(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super(CrossNet, self).__init__()
        self.hidden = 512
        self.dropout = nn.Dropout(0.2)
        
        # Pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)

        # BiLSTM for Target Encoding
        self.target_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)

        # BiLSTM for Document Encoding (Conditioned on Target)
        self.text_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, num_layers=1, dropout=0.2, bidirectional=True, batch_first=True)

        # Attention Mechanism
        self.W = nn.Linear(2 * self.hidden, self.hidden)
        self.w = nn.Linear(self.hidden, 1)
        self.act = nn.Tanh()

        # Layer Normalization (to match Keras implementation)
        self.layer_norm = nn.LayerNorm(self.hidden * 2)

        # Final Classification
        self.classifier = nn.Linear(self.hidden * 2, num_classes)

    def forward(self, x, x_len, epoch, target_word=None):
        # Embedding
        document_enc = self.embedding(x) # Shape: (batch, max_len, embedding_dim)
        target_enc = self.embedding(target_word)  # Shape: (batch, target_len, embedding_dim)

        # Target BiLSTM Encoding
        _, (target_hn, target_cn) = self.target_bilstm(target_enc)

        target_hn = torch.stack((target_hn[0], target_hn[1]), dim=0)  # (2, batch_size, hidden_size)
        target_cn = torch.stack((target_cn[0], target_cn[1]), dim=0)  # (2, batch_size, hidden_size)
        initial_states = (target_hn, target_cn)

        # Document BiLSTM Encoding (Conditioned on Target)
        last_hidden_output, _ = self.text_bilstm(document_enc, initial_states)
        last_hidden_output = self.dropout(last_hidden_output)  

        # Attention Mechanism
        coeff = self.W(last_hidden_output)
        coeff = self.act(coeff)
        coeff = self.w(coeff)  
        attn = F.softmax(coeff, dim=1)  # ✅ Apply attention over the sequence
        attn = self.dropout(attn)  

        # Apply attention weights
        output = torch.mul(last_hidden_output, attn)
        output = torch.sum(output, dim=1)  

        # Layer Normalization (✅ Match Keras)
        output = self.layer_norm(output)

        # Classification Layer
        output = self.classifier(output)

        return output  # ✅ Return raw logits (CrossEntropyLoss handles softmax)