import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim=4096, seq_len=24, num_classes=2):
        super(Transformer, self).__init__()
        self.down1 = nn.Linear(2048,384)
        self.down2 = nn.Linear(2048, 384)
        self.classification_token = nn.Parameter(torch.randn(1 ,1 , 768))
        self.embedding = nn.Linear(768, 768)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=768, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=12)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x1 = self.down1(x[:,:,0:2048].float())
        x2 = self.down2(x[:,:,2048:4096].float())
        x = torch.cat((x1, x2), dim=2)
        x = x.float() + self.embedding(x.float())
        batch_size = x.size(0)
        classification_tokens = self.classification_token.expand(batch_size, -1,-1)
        x = torch.cat((classification_tokens, x), dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x[:, 0, :]
        x = self.fc(x)
        return x