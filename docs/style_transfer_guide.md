# RAVE v2 Style Transfer Guide

This guide explains how to perform style transfer using RAVE v2 with Adaptive Instance Normalization (AdaIN).

## Overview

RAVE v2 supports real-time style transfer through AdaIN, which allows you to:
- Train a model on one style of audio
- Apply that style to any other audio in real-time
- Switch between different styles dynamically
- Preserve audio content while transferring style characteristics

## Prerequisites

1. **Trained RAVE v2 Model**: You need a model trained with AdaIN support (v3 configuration)
2. **Source Audio**: The audio you want to transform
3. **Target Style Audio**: Audio that represents the desired style
4. **Python Environment**: With PyTorch, torchaudio, and RAVE installed

## How AdaIN Style Transfer Works

### 1. Training Phase
- Train RAVE v2 with the `v3` configuration (includes AdaIN)
- Model learns to encode/decode audio effectively
- AdaIN layers are inactive during training

### 2. Inference Phase (Style Transfer)
The style transfer happens in three steps:

#### Step 1: Learn Target Statistics
```python
# Enable target learning
model.set_learn_target(True)
model.set_learn_source(False)

# Process target style audio
for chunk in target_audio_chunks:
    model.encode(chunk)
    # AdaIN learns target statistics (mean_y, std_y)
```

#### Step 2: Learn Source Statistics
```python
# Enable source learning
model.set_learn_source(True)
model.set_learn_target(False)

# Process source audio
for chunk in source_audio_chunks:
    model.encode(chunk)
    # AdaIN learns source statistics (mean_x, std_x)
```

#### Step 3: Apply Style Transfer
```python
# Disable learning, enable style transfer
model.set_learn_source(False)
model.set_learn_target(False)

# Process source audio with style transfer
output = model(source_audio)
# AdaIN applies: (source - mean_x) / std_x * std_y + mean_y
```

## Usage Examples

### Basic Style Transfer

```bash
# Train a model with AdaIN support
rave train --config v3 --db_path /path/to/target/style/dataset --out_path /model/output --name my_style_model

# Export the model for inference
rave export --run /model/output/my_style_model --streaming

# Apply style transfer
python simple_style_transfer.py \
    --model /model/output/my_style_model.ts \
    --source /path/to/source.wav \
    --target /path/to/target_style.wav \
    --output /path/to/output.wav
```

### Real-time Style Transfer

For real-time applications (Max/MSP, Pure Data), use the exported `.ts` model:

```python
import torch

# Load the scripted model
model = torch.jit.load("my_style_model.ts")

# Learn target style (do this once)
model.set_learn_target(True)
model.set_learn_source(False)
target_audio = load_audio("target_style.wav")
model.encode(target_audio)

# Learn source audio (do this for each new source)
model.set_learn_source(True)
model.set_learn_target(False)
source_audio = load_audio("source.wav")
model.encode(source_audio)

# Apply style transfer
model.set_learn_source(False)
model.set_learn_target(False)
output = model(source_audio)
```

### Batch Processing

```bash
# Process multiple source files with the same target style
python style_transfer.py \
    --model /path/to/model.ts \
    --source /path/to/source/directory \
    --target /path/to/target_style.wav \
    --output /path/to/output/directory \
    --batch
```

## Configuration Options

### Model Configurations

- **v3**: Full AdaIN support with Snake activation and descript discriminator
- **v2_small**: Optimized for timbre transfer with smaller receptive field
- **v2**: Standard configuration (no AdaIN)

### Training Parameters

```bash
# Train with AdaIN and data augmentations
rave train \
    --config v3 \
    --augment mute \
    --augment compress \
    --db_path /path/to/dataset \
    --out_path /model/output \
    --name my_model
```

### Inference Parameters

- `--num-chunks`: Number of audio chunks to use for learning statistics
- `--streaming`: Enable streaming mode for real-time processing
- `--chunk-size`: Size of audio chunks for streaming mode

## Advanced Usage

### Custom Style Transfer Workflow

```python
import torch
import torchaudio
import rave

# Load model
model = torch.jit.load("my_model.ts")

# Function to learn statistics
def learn_statistics(audio_path, is_target=True):
    audio, sr = torchaudio.load(audio_path)
    if sr != model.sr:
        audio = torchaudio.functional.resample(audio, sr, model.sr)
    
    model.set_learn_target(is_target)
    model.set_learn_source(not is_target)
    
    # Process in chunks
    chunk_size = 8192
    for i in range(0, audio.shape[-1], chunk_size):
        chunk = audio[..., i:i+chunk_size]
        if chunk.shape[-1] < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[-1]))
        model.encode(chunk.unsqueeze(0))

# Function to apply style transfer
def apply_style_transfer(source_path, output_path):
    audio, sr = torchaudio.load(source_path)
    if sr != model.sr:
        audio = torchaudio.functional.resample(audio, sr, model.sr)
    
    model.set_learn_source(False)
    model.set_learn_target(False)
    
    output = model(audio.unsqueeze(0))
    torchaudio.save(output_path, output.cpu(), model.sr)

# Usage
learn_statistics("target_style.wav", is_target=True)
learn_statistics("source.wav", is_target=False)
apply_style_transfer("source.wav", "output.wav")
```

### Real-time Processing

For real-time applications, use streaming mode:

```python
import cached_conv as cc

# Enable streaming mode
cc.use_cached_conv(True)

# Load model
model = torch.jit.load("my_model.ts")

# Process audio in real-time chunks
def process_realtime_chunk(chunk):
    # Ensure chunk is the right size
    if chunk.shape[-1] < model.receptive_field:
        chunk = torch.nn.functional.pad(chunk, (0, model.receptive_field - chunk.shape[-1]))
    
    return model(chunk.unsqueeze(0))
```

## Troubleshooting

### Common Issues

1. **Model doesn't support AdaIN**
   - Ensure you trained with `v3` configuration
   - Check if model has AdaIN layers

2. **Poor style transfer quality**
   - Increase `num_chunks` for better statistics
   - Use longer target style audio
   - Ensure target style is consistent

3. **Audio artifacts**
   - Use streaming mode for real-time processing
   - Ensure chunk size > receptive field
   - Check audio format compatibility

4. **Memory issues**
   - Reduce batch size
   - Use smaller chunk sizes
   - Process audio in smaller segments

### Debugging

```python
# Check if model supports AdaIN
def check_adain_support(model):
    for module in model.modules():
        if 'AdaptiveInstanceNormalization' in str(type(module)):
            return True
    return False

# Check model statistics
def print_model_info(model):
    print(f"Sampling rate: {model.sr}")
    print(f"Channels: {model.n_channels}")
    print(f"Receptive field: {rave.core.get_minimum_size(model)}")
    print(f"AdaIN support: {check_adain_support(model)}")
```

## Best Practices

1. **Training**
   - Use high-quality, consistent target style audio
   - Train for sufficient steps (100k+)
   - Use data augmentations for better generalization

2. **Inference**
   - Use longer target style audio for better statistics
   - Process source audio in chunks for real-time applications
   - Reset statistics between different style transfers

3. **Audio Quality**
   - Use consistent sampling rates
   - Ensure proper audio normalization
   - Test with different audio types

## Examples

### Voice Style Transfer
```bash
# Train on a specific voice style
rave train --config v3 --db_path /voice/dataset --name voice_model

# Apply to different speech
python simple_style_transfer.py \
    --model voice_model.ts \
    --source speech.wav \
    --target voice_style.wav \
    --output styled_speech.wav
```

### Instrument Style Transfer
```bash
# Train on guitar style
rave train --config v3 --db_path /guitar/dataset --name guitar_model

# Apply to piano
python simple_style_transfer.py \
    --model guitar_model.ts \
    --source piano.wav \
    --target guitar_style.wav \
    --output guitar_piano.wav
```

### Real-time Performance
```python
# Load model once
model = torch.jit.load("performance_model.ts")

# Learn target style
model.set_learn_target(True)
model.encode(target_style_audio)

# Real-time processing loop
for audio_chunk in realtime_audio_stream:
    model.set_learn_source(False)
    model.set_learn_target(False)
    output = model(audio_chunk)
    play_audio(output)
```

This guide should help you get started with style transfer using RAVE v2. The key is understanding the three-phase process: learning target statistics, learning source statistics, and applying the style transfer. 