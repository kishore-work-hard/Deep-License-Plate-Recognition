# 🚗 Deep-License-Plate-Recognition

> Because reading license plates shouldn't feel like deciphering ancient Sanskrit manuscripts

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)

## 🎯 What's This All About?

Ever tried reading a license plate from a blurry photo taken at 80 km/h during monsoon season? Yeah, it's hard. This project trains a neural network to do exactly that - automatically read Indian vehicle registration plates from images, no matter how challenging the conditions.

We're talking about handling:
- Dusty plates that look like they've been through the apocalypse
- Motion blur from speeding motorcycles
- That weird lighting at toll booths at 3 AM
- Plates with font sizes that seem to mock standardization
- The occasional bird poop obscuring crucial characters (because India)

## 🚀 Quick Start

### Installation

```bash
# Clone this bad boy
git clone https://github.com/yourusername/indian-anpr-training.git
cd indian-anpr-training

# Install dependencies (grab a coffee, this'll take a minute)
pip install torch torchvision torchaudio
pip install albumentations opencv-python pillow numpy

# Create the models directory
mkdir -p models
```

### The "I Just Want It To Work" Command

```python
python indian_anpr_training_v4.py
```

That's it. The script will automatically use our battle-tested **Accuracy Optimized Config** and start training. Grab some popcorn, this is going to take a while.

## 🏗️ Architecture Deep Dive

### The Big Picture

This isn't your grandma's CNN. We've built a hybrid beast that combines:

1. **EfficientCNNBackbone** - Extracts visual features
2. **OptimizedBiLSTM** - Understands sequential patterns
3. **CTC Loss** - Handles variable-length sequences

Think of it like a relay race:
- CNN runs the first leg, identifying edges, curves, and character shapes
- LSTM takes the baton, understanding that "KL" probably comes before numbers in Kerala plates
- CTC brings it home, decoding the sequence even when characters overlap

### Why This Architecture?

**The CNN Part:**
```
Input Image (64x256) 
    → Stem (3→64 channels)
    → Stage 1 (64→128, downsample)
    → Stage 2 (128→256, downsample)  
    → Stage 3 (256→512, downsample)
    → Global Average Pooling
    → Output (512, 1, W/8)
```

We're using residual blocks because:
- Training deep networks without them is like climbing Everest in flip-flops
- Skip connections help gradients flow backward without vanishing
- SE (Squeeze-and-Excitation) blocks let the network focus on important features

**The LSTM Part:**
A 2-layer bidirectional LSTM that reads the sequence both forwards and backwards. Why bidirectional? Because sometimes knowing what comes AFTER a character helps identify what came BEFORE. Mind = blown.

**The CTC Magic:**
CTC (Connectionist Temporal Classification) is the secret sauce that lets us:
- Handle variable-length outputs (some plates have 9 chars, some 10)
- Train without needing character-level alignment
- Deal with repetitions and blanks naturally

## 📊 Training Configurations

We've got three flavors, like Neapolitan ice cream but for neural networks:

### 1. 🏃 Speed Optimized (The "I'm Impatient" Config)

```python
config = ANPRTrainingConfigs.get_speed_optimized_config()
# epochs: 50
# batch_size: 128
# debug_samples: 10,000
```

**When to use:**
- Rapid prototyping
- Testing new ideas
- Your GPU is older than your laptop
- You want results before the heat death of the universe

**What you'll get:**
- Training finishes in a few hours
- Decent accuracy (~75-80%)
- Not production-ready but good enough for demos

### 2. 🎯 Accuracy Optimized (The "I Mean Business" Config)

```python
config = ANPRTrainingConfigs.get_accuracy_optimized_config()
# epochs: 200
# batch_size: 64
# use_full_dataset: True
```

**When to use:**
- Production deployments
- When accuracy matters more than your electricity bill
- You've got a beefy GPU with 12GB+ VRAM
- You're willing to wait 24+ hours for glory

**What you'll get:**
- Best possible accuracy (~92-95%)
- Rock-solid predictions
- A model that handles edge cases like a boss
- Bragging rights

### 3. ⚖️ Balanced (The "Goldilocks" Config)

```python
config = ANPRTrainingConfigs.get_balanced_config()
# epochs: 100
# batch_size: 96
# use_full_dataset: True
```

**When to use:**
- Most real-world scenarios
- You want good accuracy without selling a kidney
- GPU has ~8GB VRAM
- Sweet spot between speed and performance

**What you'll get:**
- Training finishes in ~12 hours
- Great accuracy (~88-92%)
- Production-ready quality
- Won't melt your GPU

## 🔧 Low-Spec Training Guide

Got a potato for a GPU? Don't worry, we've got you covered:

### If You Have 4GB VRAM or Less

```python
# Modify these in the train_optimized_model() call:
model, history = train_optimized_model(
    images_dir="path/to/images",
    labels_file="path/to/labels.txt",
    epochs=50,              # Reduce epochs
    batch_size=32,          # Smaller batches
    debug_samples=5000,     # Limit dataset size
    use_full_dataset=False
)
```

**Additional tips:**
- Set `num_workers=0` in DataLoader (line 469) - avoid multiprocessing overhead
- Reduce hidden_size from 256 to 128 in model initialization
- Use mixed precision training (add this magic):

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(images)
    loss = ctc_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### If You're CPU-Only (Brave Soul)

```python
# Absolute minimum config:
model, history = train_optimized_model(
    images_dir="path/to/images",
    labels_file="path/to/labels.txt",
    epochs=20,
    batch_size=8,           # Tiny batches
    debug_samples=1000,     # Very limited data
    use_full_dataset=False
)
```

Set `pin_memory=False` in all DataLoaders. Training will be slow (days instead of hours), but it'll work!

## 🎨 Data Augmentation Magic

We don't just throw random transformations at your data. Every augmentation is carefully chosen for Indian road conditions:

```python
# Geometric transforms
ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5)
# Handles: Camera angles, vehicle movement

Perspective(scale=(0.02, 0.08))  
# Handles: Viewing angles, mounted cameras

# Photometric transforms
RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
# Handles: Day/night, shadows, toll booth lighting

RandomGamma(gamma_limit=(85, 115))
# Handles: Washed-out plates, underexposed images

GaussNoise(var_limit=(5, 15))
# Handles: Low-quality cameras, sensor noise

MotionBlur(blur_limit=3)
# Handles: Moving vehicles, shaky cameras
```

## 📈 Understanding the Metrics

When training, you'll see two key accuracy metrics:

### Sequence Accuracy (The Strict Teacher)

```
Sequence Accuracy: 87.5%
```

This means 87.5% of plates were read PERFECTLY. Every single character correct. It's harsh but honest.

**What's good?**
- 90%+ : Production ready
- 80-90% : Pretty good, needs fine-tuning
- 70-80% : Getting there, keep training
- <70% : Houston, we have a problem

### Character Accuracy (The Encouraging Coach)

```
Character Accuracy: 94.2%
```

This measures individual character correctness. If the model reads "KL07AB1234" as "KL07AB1234" (wrong last digit), sequence accuracy is 0% but character accuracy is 90%.

**Why both matter?**
- High character accuracy + low sequence accuracy = Almost there, just needs more training
- Low both = Check your data quality, something's wrong
- High both = 🎉 You did it!

## 🐛 Troubleshooting

### "My training is stuck at 0% accuracy!"

**Possible causes:**
1. **Labels don't match images**: Check your `rec_train.txt` file format
2. **Bad data**: Corrupted images or mislabeled plates
3. **Learning rate too high**: Try reducing max_lr in OneCycleLR
4. **Not enough epochs**: Some datasets are stubborn, give it time

**Quick fix:**
```python
# Add this validation before training:
train_dataset = FastIndianDataset(...)
for i in range(5):
    img, target, length = train_dataset[i]
    print(f"Sample {i}: Length={length}, Target shape={target.shape}")
```

### "CUDA Out of Memory"

You know this one. Here's the priority fix list:
1. Reduce batch_size (try 32, then 16, then 8)
2. Reduce debug_samples to 5000
3. Set num_workers=0
4. Reduce hidden_size to 128
5. Clear cache: `torch.cuda.empty_cache()`

### "Training is too slow!"

**On GPU:**
- Increase batch_size (more parallelization)
- Ensure `pin_memory=True` in DataLoaders
- Update GPU drivers
- Close other GPU-hungry apps (looking at you, Chrome)

**On CPU:**
- This is your life now. Embrace the slowness or rent a GPU

### "Model predicts gibberish"

```
Sample predictions:
  'AAAAAAAAAA'
  'KLKLKLKLKL'  
  'AAAAAA1111'
```

This is the neural network equivalent of a toddler babbling. Usually means:
- Not enough training (increase epochs)
- Learning rate is off (try reducing by 50%)
- Data quality issues (check your labels)
- Model capacity too small (increase hidden_size)

## 📁 File Structure

```
indian-anpr-training/
├── indian_anpr_training_v4.py    # Main training script
├── models/                        # Saved model checkpoints
│   ├── best_sequence_accuracy_anpr.pth
│   └── best_character_accuracy_anpr.pth
├── create_dataset/
│   └── det/
│       └── output/
│           ├── rec_images/        # Your plate images
│           └── rec_train.txt      # Labels file
└── README.md                      # You are here!
```

## 🎓 How Training Works

### The Training Loop (Simplified)

```
For each epoch:
  For each batch of images:
    1. Load images and labels
    2. Apply augmentations (make it harder)
    3. Feed through CNN → extract features
    4. Feed through LSTM → understand sequence
    5. Calculate CTC loss (how wrong are we?)
    6. Backpropagate (learn from mistakes)
    7. Update weights (get smarter)
  
  Validate on unseen data:
    - Check sequence accuracy
    - Check character accuracy
    - Save if best model so far
```

### Learning Rate Scheduling

We use **OneCycleLR** - the fancy pants of learning rate schedules:

```
Learning Rate over time:

  0.003 |     ***
        |    *   *
  0.002 |   *     *
        |  *       *
  0.001 | *         *
        |*           ***
  0.0001|________________***********
        0%  10%  20%  ...  90%  100%
        Warm-up  Peak  Annealing
```

**Why this shape?**
- **Warm-up**: Gently wake up the network, don't shock it
- **Peak**: Let it learn aggressively
- **Annealing**: Fine-tune, polish the rough edges

## 🔬 Advanced Customization

### Tweaking the Model Architecture

Want to experiment? Here's what you can modify:

```python
# In EfficientANPRModel.__init__()

# Smaller model (faster, less accurate):
hidden_size=128  # Default: 256

# Deeper LSTM (more sequence understanding):
num_layers=3  # Default: 2

# More channels (better feature extraction):
# Modify stage channels in EfficientCNNBackbone
self.stage3 = self._make_stage(256, 768, 2, stride=2)  # Default: 512
```

### Custom Augmentation Pipeline

```python
# In OptimizedAugmentation.__init__()

# For cleaner plates (reduce noise):
A.GaussNoise(var_limit=(2, 8), p=0.1)  # Less noise

# For dirtier plates (more variation):
A.RandomRain(slant_range=(-10, 10), p=0.2)  # Add rain effects
A.RandomShadow(p=0.3)  # Add shadow simulation
```

### Hyperparameter Tuning Guide

| Parameter | Low-End | Balanced | High-End | Effect |
|-----------|---------|----------|----------|--------|
| batch_size | 16-32 | 64-96 | 128-256 | Larger = faster but needs more VRAM |
| hidden_size | 128 | 256 | 512 | Larger = more capacity but slower |
| epochs | 20-50 | 100 | 200+ | More = better accuracy but time |
| max_lr | 0.001 | 0.003 | 0.005 | Higher = faster learning but risky |
| weight_decay | 0.01 | 0.02 | 0.05 | Higher = more regularization |

## 📊 Expected Results

### Timeline (on a decent GPU - RTX 3060 or better):

| Config | Time | Seq. Acc. | Char. Acc. | Use Case |
|--------|------|-----------|-----------|----------|
| Speed Optimized | 3-5 hours | 75-80% | 85-88% | Prototyping |
| Balanced | 10-14 hours | 88-92% | 93-96% | Production |
| Accuracy Optimized | 24-30 hours | 92-95% | 96-98% | Mission Critical |

### Sample Training Output:

```
📈 Epoch 45/100
------------------------------------------------------------
  Batch    0/245, Loss: 0.8234, LR: 0.002845
  Batch   50/245, Loss: 0.6521, LR: 0.002831
  Batch  100/245, Loss: 0.5892, LR: 0.002816

🔍 Validation Phase...

📊 Epoch 45 Results:
  Training Loss:     0.6215
  Validation Loss:   0.5834
  Sequence Accuracy: 89.3% (Exact plate matches)
  Character Accuracy: 94.7% (Individual character accuracy)

📝 Sample Predictions:
  Sample 1: 'KL07AB1234'
  Sample 2: 'MH12CD5678'
  Sample 3: 'DL04EF9012'

✅ Best sequence accuracy model saved! 89.3%
------------------------------------------------------------
```

## 🎯 Performance Optimization Tips

### GPU Utilization

Monitor your GPU usage:
```bash
watch -n 1 nvidia-smi
```

**Ideal numbers:**
- GPU Utilization: 95-100%
- Memory Usage: 80-95% of available
- Temperature: <85°C

**If GPU usage is low (<70%):**
- Increase batch_size
- Increase num_workers in DataLoader
- Check if CPU is bottlenecking (is it at 100%?)

### CPU Bottleneck Fixes

```python
# Increase parallel data loading
num_workers=4  # Try 4, 6, or 8

# Enable persistent workers (keeps them alive between epochs)
persistent_workers=True

# Use faster image loading
# Replace PIL with cv2 (already done in this version!)
```

## 🤝 Contributing

Found a bug? Have a cool idea? PRs welcome! 

**Good first issues:**
- Add support for multi-line plates
- Implement focal loss for hard examples
- Add TensorBoard logging
- Create inference script
- Add model quantization for edge devices

## 📝 License

MIT License - Go wild, just don't sue us if your ANPR system can't read a plate during a apocalyptic dust storm.

## 🙏 Acknowledgments

- PyTorch team for making deep learning accessible
- Albumentations for the augmentation library
- Indian Motor Vehicles Department for the interesting plate format variations
- Caffeine, for obvious reasons

## 📞 Support

- **Issues**: Open a GitHub issue
- **Questions**: Start a discussion
- **Emergency**: Light the Bat-Signal (or just open an issue)

---

Made with ☕ and 🧠 by developers who've spent way too much time looking at license plates

**Remember**: The journey of a thousand miles begins with a single `git clone`. Now go train some networks!
