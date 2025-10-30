"""
Mobile Model Export for Indian ANPR 
====================================================
Export trained PyTorch models to mobile-friendly formats with proper compatibility fixes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import copy

# Import your model architecture
from indian_anpr_training_v4 import (
    EfficientANPRModel, 
    EfficientCNNBackbone,
    OptimizedBiLSTM,
    OptimizedResBlock,
    LightweightSEBlock
)


class MobileCompatibleCNNBackbone(nn.Module):
    """
    ONNX-compatible CNN backbone - replaces AdaptiveAvgPool2d with fixed pooling
    """
    def __init__(self, img_height=64):
        super(MobileCompatibleCNNBackbone, self).__init__()
        
        # Streamlined initial layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Efficient residual stages
        self.stage1 = self._make_stage(64, 128, 2, stride=2)
        self.stage2 = self._make_stage(128, 256, 2, stride=2)
        self.stage3 = self._make_stage(256, 512, 2, stride=2)
        
        # Simplified output processing
        self.final_conv = nn.Conv2d(512, 512, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(512)
        
        # FIXED: Use fixed-size pooling instead of AdaptiveAvgPool2d
        # For img_height=64, after 4 downsamples (2^4=16), height=4
        # We want to pool height to 1, so kernel_size=(4, 1)
        self.pool = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = [OptimizedResBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(OptimizedResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)           # [B, 64, H/2, W]
        x = self.stage1(x)         # [B, 128, H/4, W/2]
        x = self.stage2(x)         # [B, 256, H/8, W/4]
        x = self.stage3(x)         # [B, 512, H/16, W/8]
        
        x = F.relu(self.final_bn(self.final_conv(x)), inplace=True)
        x = self.pool(x)           # [B, 512, 1, W/8]
        
        return x


class MobileCompatibleANPRModel(nn.Module):
    """
    Mobile-compatible ANPR model with fixed architecture for ONNX export
    """
    def __init__(self, num_classes, img_height=64, hidden_size=256):
        super(MobileCompatibleANPRModel, self).__init__()
        
        self.backbone = MobileCompatibleCNNBackbone(img_height)
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_height, 256)
            backbone_out = self.backbone(dummy_input)
            self.feature_dim = backbone_out.size(1)
        
        # Simplified sequence model
        self.sequence_model = OptimizedBiLSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=2
        )
        
        # Streamlined classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        features = self.backbone(x)  # [B, 512, 1, W]
        
        # Reshape for LSTM
        batch_size, channels, height, width = features.size()
        features = features.squeeze(2).permute(0, 2, 1)  # [B, W, 512]
        
        # Sequence modeling
        sequence_out = self.sequence_model(features)  # [B, W, hidden_size*2]
        
        # Classification
        output = self.classifier(sequence_out)  # [B, W, num_classes]
        
        # For CTC: [W, B, num_classes]
        return output.permute(1, 0, 2)


def convert_trained_to_mobile_compatible(checkpoint_path):
    """
    Convert trained model to mobile-compatible version
    """
    print("🔄 Converting trained model to mobile-compatible version...")
    
    # Load original checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('model_config', {})
    
    # Create mobile-compatible model
    mobile_model = MobileCompatibleANPRModel(
        num_classes=config.get('num_classes', 37),
        img_height=config.get('img_height', 64),
        hidden_size=config.get('hidden_size', 256)
    )
    
    # Load original model
    original_model = EfficientANPRModel(
        num_classes=config.get('num_classes', 37),
        img_height=config.get('img_height', 64),
        hidden_size=config.get('hidden_size', 256)
    )
    original_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Transfer weights
    print("📦 Transferring weights...")
    
    # Copy all weights except the problematic pooling layer
    mobile_state = mobile_model.state_dict()
    original_state = original_model.state_dict()
    
    for name, param in original_state.items():
        if name in mobile_state and mobile_state[name].shape == param.shape:
            mobile_state[name].copy_(param)
    
    mobile_model.load_state_dict(mobile_state)
    
    print("✅ Conversion successful!")
    
    # Verify outputs match
    mobile_model.eval()
    original_model.eval()
    
    test_input = torch.randn(1, 3, 64, 256)
    with torch.no_grad():
        original_out = original_model(test_input)
        mobile_out = mobile_model(test_input)
        diff = torch.abs(original_out - mobile_out).max().item()
        print(f"🔍 Max difference between models: {diff:.6f}")
    
    return mobile_model, checkpoint


class MobileANPRExporter:
    """
    Comprehensive model exporter for mobile deployment
    """
    def __init__(self, checkpoint_path, output_dir='./mobile_models'):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and convert model
        print(f"📦 Loading checkpoint from {checkpoint_path}")
        self.mobile_model, self.checkpoint = convert_trained_to_mobile_compatible(checkpoint_path)
        
        # Extract model config
        self.config = self.checkpoint.get('model_config', {})
        self.char_to_idx = self.checkpoint['char_to_idx']
        self.idx_to_char = self.checkpoint['idx_to_char']
        
        print(f"✅ Model loaded successfully")
        print(f"   Sequence Accuracy: {self.checkpoint.get('seq_accuracy', 0):.1%}")
        print(f"   Character Accuracy: {self.checkpoint.get('char_accuracy', 0):.1%}")
        
    def export_to_onnx(self, opset_version=12, simplify_model=True):
        """
        Export to ONNX format - Universal format for mobile/edge
        """
        print("\n🔄 Exporting to ONNX format...")
        
        self.mobile_model.eval()
        
        # Create dummy input
        img_height = self.config.get('img_height', 64)
        dummy_input = torch.randn(1, 3, img_height, 256)
        
        onnx_path = os.path.join(self.output_dir, 'anpr_model.onnx')
        
        # Export with optimization
        torch.onnx.export(
            self.mobile_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {1: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model exported: {onnx_path}")
        
        # Simplify ONNX model
        if simplify_model:
            print("🔧 Simplifying ONNX model...")
            try:
                import onnx
                from onnxsim import simplify
                
                model_onnx = onnx.load(onnx_path)
                model_simp, check = simplify(model_onnx)
                
                if check:
                    simplified_path = os.path.join(self.output_dir, 'anpr_model_simplified.onnx')
                    onnx.save(model_simp, simplified_path)
                    print(f"✅ Simplified ONNX saved: {simplified_path}")
                    
                    # Compare sizes
                    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
                    simplified_size = os.path.getsize(simplified_path) / (1024 * 1024)
                    print(f"   Original: {original_size:.2f} MB")
                    print(f"   Simplified: {simplified_size:.2f} MB")
                    print(f"   Reduction: {(1 - simplified_size/original_size)*100:.1f}%")
            except Exception as e:
                print(f"⚠️  Simplification failed: {e}")
        
        return onnx_path
    
    def export_to_torchscript(self):
        """
        Export to TorchScript - Native PyTorch mobile format
        """
        print("\n🔄 Exporting to TorchScript format...")
        
        self.mobile_model.eval()
        
        # Create dummy input
        img_height = self.config.get('img_height', 64)
        dummy_input = torch.randn(1, 3, img_height, 256)
        
        # Trace the model
        traced_model = torch.jit.trace(self.mobile_model, dummy_input)
        
        # Optimize for mobile
        traced_model_optimized = torch.jit.optimize_for_mobile(traced_model)
        
        torchscript_path = os.path.join(self.output_dir, 'anpr_model.pt')
        traced_model_optimized._save_for_lite_interpreter(torchscript_path)
        
        print(f"✅ TorchScript model exported: {torchscript_path}")
        
        file_size = os.path.getsize(torchscript_path) / (1024 * 1024)
        print(f"   File size: {file_size:.2f} MB")
        
        return torchscript_path
    
    def save_metadata(self):
        """Save model metadata for mobile inference"""
        metadata = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'num_classes': self.config.get('num_classes'),
            'img_height': self.config.get('img_height', 64),
            'img_width': 256,
            'hidden_size': self.config.get('hidden_size', 256),
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'sequence_accuracy': self.checkpoint.get('seq_accuracy', 0),
            'character_accuracy': self.checkpoint.get('char_accuracy', 0)
        }
        
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Metadata saved: {metadata_path}")
        return metadata_path
    
    def export_for_android(self):
        """
        Android-optimized export (ONNX + TorchScript)
        """
        print("📱 Exporting for Android deployment...\n")
        print("=" * 60)
        
        results = {}
        
        # Export to ONNX (primary)
        try:
            results['onnx'] = self.export_to_onnx()
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
        
        # Export to TorchScript (alternative)
        try:
            results['torchscript'] = self.export_to_torchscript()
        except Exception as e:
            print(f"❌ TorchScript export failed: {e}")
        
        # Save metadata
        results['metadata'] = self.save_metadata()
        
        print("\n" + "=" * 60)
        print("✅ Android export completed!\n")
        
        # Summary
        print("📊 Export Summary:")
        print("-" * 60)
        for format_name, path in results.items():
            if path and os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"✓ {format_name.upper():15s}: {size:6.2f} MB - {path}")
        
        print("\n📝 Integration Guide:")
        print("-" * 60)
        print("Using ONNX Runtime (Recommended):")
        print("  1. Add dependency: implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'")
        print("  2. Copy anpr_model_simplified.onnx to app/src/main/assets/")
        print("  3. Copy model_metadata.json to app/src/main/assets/")
        print("")
        print("Using PyTorch Mobile:")
        print("  1. Add dependency: implementation 'org.pytorch:pytorch_android:1.13.1'")
        print("  2. Copy anpr_model.pt to app/src/main/assets/")
        print("  3. Copy model_metadata.json to app/src/main/assets/")
        
        return results


def verify_onnx_model(onnx_path):
    """
    Verify ONNX model works correctly
    """
    print(f"\n🔍 Verifying ONNX model: {onnx_path}")
    
    try:
        import onnxruntime as ort
        
        # Load model
        session = ort.InferenceSession(onnx_path)
        
        # Check inputs/outputs
        print("\n📥 Model Inputs:")
        for inp in session.get_inputs():
            print(f"   {inp.name}: {inp.shape} ({inp.type})")
        
        print("\n📤 Model Outputs:")
        for out in session.get_outputs():
            print(f"   {out.name}: {out.shape} ({out.type})")
        
        # Test inference
        dummy_input = np.random.randn(1, 3, 64, 256).astype(np.float32)
        outputs = session.run(None, {'input': dummy_input})
        
        print(f"\n✅ Inference test passed!")
        print(f"   Output shape: {outputs[0].shape}")
        
        return True
    except ImportError:
        print("⚠️  Install onnxruntime for verification: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export ANPR model for mobile deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./mobile_models',
                       help='Output directory for exported models')
    parser.add_argument('--platform', type=str, default='android',
                       choices=['android', 'verify'],
                       help='Target platform or action')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = MobileANPRExporter(args.checkpoint, args.output_dir)
    
    if args.platform == 'android':
        # Export for Android
        results = exporter.export_for_android()
        
        # Verify ONNX if available
        if 'onnx' in results and results['onnx']:
            verify_onnx_model(results['onnx'])
            
    elif args.platform == 'verify':
        # Just verify existing ONNX
        onnx_path = os.path.join(args.output_dir, 'anpr_model.onnx')
        if os.path.exists(onnx_path):
            verify_onnx_model(onnx_path)
        else:
            print(f"❌ ONNX model not found at {onnx_path}")
    
    print("\n✅ All done!")
