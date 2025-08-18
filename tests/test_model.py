import unittest
import torch
import numpy as np
from torchinfo import summary

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import BengaliResNetOCR, BengaliViTOCR, BengaliHybridOCR
from configs.config import Config

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_channels = 1
        self.img_height = Config.IMG_HEIGHT
        self.img_width = Config.IMG_WIDTH
        self.num_classes = len(Config.BENGALI_CHARS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dummy_input = torch.randn(
            self.batch_size, 
            self.num_channels, 
            self.img_height, 
            self.img_width
        ).to(self.device)
        
        self.resnet_seq_len = 256
        self.vit_seq_len = 196  
        self.hybrid_seq_len = 452 

    def test_resnet_model(self):
        model = BengaliResNetOCR(self.num_classes).to(self.device)
        model.eval()
        
        summary(model, 
                input_size=(self.batch_size,
                            self.num_channels, 
                            self.img_height, 
                            self.img_width), 
                verbose=0)
        
        with torch.no_grad():
            output = model(self.dummy_input)
            
        self.assertEqual(output.shape, (self.resnet_seq_len, self.batch_size, self.num_classes),
                         "ResNet output shape mismatch")
        
        self.assertTrue(hasattr(model, 'feature_extractor'), 
                        "ResNet model missing feature_extractor")
        self.assertTrue(hasattr(model, 'adapt_pool'), 
                        "ResNet model missing adapt_pool")
        self.assertTrue(hasattr(model, 'gru'), 
                        "ResNet model missing GRU layer")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nResNet Model Parameters: {total_params:,}")
        self.assertGreater(total_params, 10_000_000, 
                           "ResNet model has too few parameters")
        
    def test_vit_model(self):
        model = BengaliViTOCR(self.num_classes).to(self.device)
        model.eval()
        
        summary(model, input_size=(self.batch_size, 
                                self.num_channels, 
                                self.img_height, 
                                self.img_width), 
                verbose=0)
        
        with torch.no_grad():
            output = model(self.dummy_input)
            
        self.assertEqual(output.shape, (self.vit_seq_len, self.batch_size, self.num_classes),
                         "ViT output shape mismatch")
        
        self.assertTrue(hasattr(model, 'vit'), 
                        "ViT model missing vit module")
        self.assertTrue(hasattr(model, 'feature_reducer'), 
                        "ViT model missing feature_reducer")
        self.assertTrue(hasattr(model, 'gru'), 
                        "ViT model missing GRU layer")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nViT Model Parameters: {total_params:,}")
        self.assertGreater(total_params, 80_000_000, 
                           "ViT model has too few parameters")
        
    def test_hybrid_model(self):
        model = BengaliHybridOCR(self.num_classes).to(self.device)
        model.eval()
        
        summary(model, input_size=(self.batch_size, 
                                   self.num_channels, 
                                   self.img_height, 
                                   self.img_width), 
                verbose=0)
        
        with torch.no_grad():
            output = model(self.dummy_input)
            
        self.assertEqual(output.shape, (self.hybrid_seq_len, self.batch_size, self.num_classes),
                         "Hybrid output shape mismatch")
        
        self.assertTrue(hasattr(model, 'cnn_extractor'), 
                        "Hybrid model missing cnn_extractor")
        self.assertTrue(hasattr(model, 'vit'), 
                        "Hybrid model missing vit module")
        self.assertTrue(hasattr(model, 'gru'), 
                        "Hybrid model missing GRU layer")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nHybrid Model Parameters: {total_params:,}")
        self.assertGreater(total_params, 100_000_000, 
                           "Hybrid model has too few parameters")
        
    def test_model_outputs(self):
        models = {
            "ResNet": BengaliResNetOCR(self.num_classes),
            "ViT": BengaliViTOCR(self.num_classes),
            "Hybrid": BengaliHybridOCR(self.num_classes)
        }
        
        for name, model in models.items():
            model = model.to(self.device).eval()
            with torch.no_grad():
                output = model(self.dummy_input)
                
            self.assertFalse(torch.isnan(output).any(), 
                             f"{name} model output contains NaN values")
            
            self.assertFalse(torch.isinf(output).any(), 
                             f"{name} model output contains infinite values")
 
            self.assertTrue((output > -100).all() and (output < 100).all(),
                            f"{name} model output has extreme values")
            
    def test_model_device_transfer(self):
        models = [
            BengaliResNetOCR(self.num_classes),
            BengaliViTOCR(self.num_classes),
            BengaliHybridOCR(self.num_classes)
        ]
        
        for model in models:
            model.to("cpu")
            output = model(torch.randn(self.batch_size, self.num_channels, 
                            self.img_height, self.img_width))
            self.assertEqual(output.device.type, "cpu",
                            "Model not moved to CPU correctly")

            if torch.cuda.is_available():
                model.to("cuda")
                output = model(torch.randn(self.batch_size, self.num_channels, 
                                    self.img_height, self.img_width).to("cuda"))
                self.assertEqual(output.device.type, "cuda",
                                "Model not moved to GPU correctly")
                
    def test_model_saving_loading(self):
        models = {
            "resnet": BengaliResNetOCR(self.num_classes),
            "vit": BengaliViTOCR(self.num_classes),
            "hybrid": BengaliHybridOCR(self.num_classes)
        }
        
        for name, model in models.items():
            model_path = f"test_{name}_model.pth"
            torch.save(model.state_dict(), model_path)

            loaded_model = type(model)(self.num_classes)
            loaded_model.load_state_dict(torch.load(model_path))

            os.remove(model_path)
            
            for orig_param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(orig_param, loaded_param),
                                f"Parameters mismatch in {name} model after loading")
                
    def test_model_training_mode(self):
        models = [
            BengaliResNetOCR(self.num_classes),
            BengaliViTOCR(self.num_classes),
            BengaliHybridOCR(self.num_classes)
        ]
        
        for model in models:
            model.train()
            for param in model.parameters():
                self.assertTrue(param.requires_grad, 
                                "Parameters should require grad in train mode")
            self.assertTrue(model.training, "Model should be in training mode")
            
            model.eval()
            for param in model.parameters():
                self.assertTrue(param.requires_grad, 
                                "Parameters should still require grad in eval mode")
            self.assertFalse(model.training, "Model should be in eval mode")

if __name__ == "__main__":
    unittest.main(verbosity=2)