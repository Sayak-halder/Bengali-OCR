import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import configs.config as config
from transformers import ViTModel


class BengaliResNetOCR(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
            self.backbone.maxpool, self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4
        )
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 32))
        self.feature_reducer = nn.Conv2d(512, 256, 1)
        self.gru = nn.GRU(256, config.HIDDEN_SIZE, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(config.HIDDEN_SIZE * 2, num_chars)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adapt_pool(features)
        features = self.feature_reducer(features)
        B, C, H, W = features.size()
        features = features.view(B, C, -1).permute(0, 2, 1)
        gru_out, _ = self.gru(features)
        return self.classifier(gru_out).permute(1, 0, 2)
    

class BengaliViTOCR(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            3, 768, kernel_size=16, stride=16
        )
        
        with torch.no_grad():
            pretrained_conv = ViTModel.from_pretrained("google/vit-base-patch16-224"
                ).embeddings.patch_embeddings.projection
            new_weight = pretrained_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            self.vit.embeddings.patch_embeddings.projection.weight.data = new_weight
            self.vit.embeddings.patch_embeddings.projection.bias.data = pretrained_conv.bias.data

        self.target_seq_len = 32
        self.feature_reducer = nn.Linear(768, config.HIDDEN_SIZE)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.target_seq_len, config.HIDDEN_SIZE))
        self.gru = nn.GRU(config.HIDDEN_SIZE, config.HIDDEN_SIZE, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(config.HIDDEN_SIZE * 2, num_chars)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        
        vit_out = self.vit(pixel_values=x).last_hidden_state
        patch_features = vit_out[:, 1:]

        features = self.feature_reducer(patch_features)
        if features.size(1) > self.target_seq_len:
            features = features.permute(0, 2, 1)
            features = F.interpolate(features, size=self.target_seq_len, mode='linear', align_corners=False)
            features = features.permute(0, 2, 1)
        elif features.size(1) < self.target_seq_len:
            pad_size = self.target_seq_len - features.size(1)
            features = F.pad(features, (0, 0, 0, pad_size), "constant", 0)
        
        features = features + self.position_embeddings
        gru_out, _ = self.gru(features)
        return self.classifier(gru_out).permute(1, 0, 2)
    

class BengaliHybridOCR(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn_extractor = nn.Sequential(
            self.cnn.conv1, self.cnn.bn1, self.cnn.relu,
            self.cnn.maxpool, self.cnn.layer1, self.cnn.layer2
        )
        
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        with torch.no_grad():
            pretrained_conv = ViTModel.from_pretrained("google/vit-base-patch16-224"
                ).embeddings.patch_embeddings.projection
            new_weight = pretrained_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            self.vit.embeddings.patch_embeddings.projection.weight.data = new_weight
            self.vit.embeddings.patch_embeddings.projection.bias.data = pretrained_conv.bias.data
        
        self.target_seq_len = 32
        
        self.cnn_reducer = nn.Conv2d(128, 256, 1)
        self.cnn_adaptive_pool = nn.AdaptiveAvgPool2d((1, self.target_seq_len))

        self.vit_reducer = nn.Linear(768, 256)
        self.vit_adaptive_pool = nn.AdaptiveAvgPool1d(self.target_seq_len)

        self.gru = nn.GRU(512, config.HIDDEN_SIZE, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(config.HIDDEN_SIZE * 2, num_chars)

    def forward(self, x):
        cnn_features = self.cnn_extractor(x) 
        cnn_features = self.cnn_reducer(cnn_features)
        cnn_features = self.cnn_adaptive_pool(cnn_features)
        cnn_features = cnn_features.squeeze(2).permute(0, 2, 1)

        x_vit = x.repeat(1, 3, 1, 1)
        x_vit = F.interpolate(x_vit, size=(224, 224), mode="bilinear")
        vit_features = self.vit(pixel_values=x_vit).last_hidden_state
        vit_features = vit_features[:, 1:, :]
        vit_features = self.vit_reducer(vit_features)
        vit_features = vit_features.permute(0, 2, 1)
        vit_features = self.vit_adaptive_pool(vit_features)
        vit_features = vit_features.permute(0, 2, 1)

        combined = torch.cat([cnn_features, vit_features], dim=-1)

        gru_out, _ = self.gru(combined)
        output = self.classifier(gru_out)
        return output.permute(1, 0, 2)   