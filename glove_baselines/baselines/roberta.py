

from torch import nn

class RobertaModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(RobertaModel, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        # Dynamically set the output size based on the number of classes
        self.fc2 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
        # Softmax is not applied here if using nn.CrossEntropyLoss
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        # No softmax applied in forward if using nn.CrossEntropyLoss
        return x