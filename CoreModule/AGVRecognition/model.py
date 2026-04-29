import torch
import torch.nn as nn
import torch.nn.functional as F
class TrainablePositionEncoding(nn.Module):# 位置编码
    def __init__(self, max_sequence_length, d_model):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
        self.pe_embedding = nn.Embedding(max_sequence_length, d_model)
        nn.init.constant_(self.pe_embedding.weight, 0.) 

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()
        positions = torch.arange(0, sequence_length).unsqueeze(0).expand(batch_size, sequence_length).to(x.device)
        pe = self.pe_embedding(positions)
        x = x + pe
        return x



class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.transformer_encoder_layers =  nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layers, num_layers)

    def forward(self, x):
        output = self.transformer_encoder(x.float() )
        return output

class ClassifierHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        batch_size, sequence_length, d_model = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class TwoTransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, num_classes):
        super(TwoTransformerClassifier, self).__init__()
        self.transformerModel = TransformerModel(input_size, hidden_size, num_layers, num_heads, dropout)
        self.classifier_head = ClassifierHead(hidden_size*8, hidden_size* 2, num_classes)
    def forward(self, x):
        output = self.transformerModel(x)
        output = self.classifier_head(output)
        return output


