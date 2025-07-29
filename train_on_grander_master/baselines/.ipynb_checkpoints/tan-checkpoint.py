import torch
import torch.nn as nn
import torch.nn.functional as F


# class TAN(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(TAN, self).__init__()

#         self.hidden = 512
#         self.dropout = nn.Dropout(0.2)
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)
#         self.birnn = nn.RNN(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)
#         self.linear = nn.Linear(2 * embedding_matrix.shape[1], 1)
#         self.classifier = nn.Linear(self.hidden * 2, num_classes)

#     def forward(self, x, x_len, epoch, target_word=None):
#         # embedding layer
#         document_enc = self.embedding(x)  # batch maxlen glove-hidden
#         target_enc = self.embedding(target_word)

#         # Bi-RNN
#         last_hidden_output, _ = self.birnn(document_enc)  # document-encoding
#         last_hidden_output = self.dropout(last_hidden_output)

#         # Target-augmented embedding
#         target_enc = torch.mean(target_enc, dim=1)  # [batchsize, n_words, hidden] => [batchsize, hidden]
#         target_enc = target_enc.unsqueeze(1)
#         target_enc = target_enc.expand(target_enc.shape[0], document_enc.shape[1], target_enc.shape[2])
#         aug_emb = torch.cat([document_enc, target_enc], dim=-1)  # [batchsize, n_words, 2*hidden]

#         # Linear
#         context_emb = self.linear(aug_emb)
#         context_score = F.softmax(context_emb, dim=-1)
#         # print('context', context_score.shape) [batchsize words 1]

#         # Inner Product
#         output = torch.mul(last_hidden_output, context_score)
#         # print(output.shape) [batchsize, words, hidden]

#         # classification
#         output = torch.sum(output, dim=1)  # [batchsize, 2*hidden]
#         output = self.classifier(output)

#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F

# class TAN(nn.Module):
#     def __init__(self, embedding_matrix, num_classes):
#         super(TAN, self).__init__()

#         self.hidden = 512
#         self.dropout = nn.Dropout(0.2)
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(), freeze=True)

#         # BiLSTM for Sentence Encoding
#         self.bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)

#         # BiLSTM for Target Encoding
#         self.target_bilstm = nn.LSTM(embedding_matrix.shape[1], self.hidden, bidirectional=True, batch_first=True)

#         # Attention Weights
#         self.W_h = nn.Parameter(torch.rand([2 * self.hidden, 1], requires_grad=True))

#         # Fully Connected Layers
#         self.linear = nn.Linear(2 * self.hidden, 256)
#         self.classifier = nn.Linear(256, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x, x_len, epoch, target_word):
#         """
#         x: input sentence tensor (batch_size, max_sentence_len)
#         x_len: list containing lengths of each sentence in the batch
#         target_word: target tensor (batch_size, max_target_len)
#         """
#         # Convert x_len to tensor
#         seq_lengths = torch.tensor(x_len, dtype=torch.long, device=x.device)  # ✅ Fix

#         # Embedding Layer
#         document_enc = self.embedding(x)
#         target_enc = self.embedding(target_word)

#         # **Sort input by length for PackedSequence processing**
#         seq_lengths, perm_idx = seq_lengths.sort(descending=True)  # ✅ Fix
#         document_enc = document_enc[perm_idx]  # Reorder input based on length

#         # **Pack padded sequences dynamically**
#         packed_x = nn.utils.rnn.pack_padded_sequence(document_enc, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

#         # **Pass through BiLSTM**
#         packed_output, (ht, ct) = self.bilstm(packed_x)

#         # **Unpack sequences after processing**
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

#         # **Restore original order of batch**
#         _, unperm_idx = perm_idx.sort()
#         h_lstm = output[unperm_idx]

#         # Target BiLSTM Encoding
#         _, (target_hn, _) = self.target_bilstm(target_enc)
#         target_enc = torch.cat((target_hn[0], target_hn[1]), dim=-1)  # Combine BiLSTM forward & backward states
#         target_enc = target_enc.unsqueeze(1)  # Reshape to (batch_size, 1, hidden_size * 2)

#         # Target-Aware Attention
#         attn_weights = torch.matmul(h_lstm, self.W_h)
#         attn_weights = F.softmax(attn_weights, dim=1)  # Normalize attention over sequence length
#         context_vector = torch.sum(h_lstm * attn_weights, dim=1)  # Weighted sum

#         # Dropout & Classification
#         context_vector = self.dropout(context_vector)
#         linear = self.relu(self.linear(context_vector))
#         linear = self.dropout(linear)
#         output = self.classifier(linear)

#         return output  # Raw logits (CrossEntropyLoss applies softmax)


class TAN(nn.Module):
    def __init__(self, num_classes):
        super(TAN, self).__init__()

        self.hidden_size = 512  # LSTM hidden size
        self.input_size = 384  # BERT embedding dimension
        self.dropout = nn.Dropout(0.2)

        # BiLSTM for Sentence Encoding
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)

        # Attention Weights
        self.W_h = nn.Parameter(torch.rand([2 * self.hidden_size, 1], requires_grad=True))

        # Fully Connected Layers
        self.linear = nn.Linear(2 * self.hidden_size, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text_emb):
        """
        text_emb: (batch_size, 384) -> Precomputed BERT embeddings
        """

        # Reshape input for BiLSTM (batch, seq_len=1, embedding_dim=384)
        x = text_emb.unsqueeze(1)  # Shape: (batch, 1, 384)

        # Pass through BiLSTM
        h_lstm, _ = self.bilstm(x)  # Shape: (batch, 1, 2*hidden_size)

        # Attention Mechanism
        attn_weights = torch.matmul(h_lstm, self.W_h)  # Compute attention scores
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize over sequence length
        context_vector = torch.sum(h_lstm * attn_weights, dim=1)  # Weighted sum

        # Dropout & Classification
        context_vector = self.dropout(context_vector)
        linear = self.relu(self.linear(context_vector))
        linear = self.dropout(linear)
        output = self.classifier(linear)

        return output  # Raw logits (CrossEntropyLoss applies softmax)