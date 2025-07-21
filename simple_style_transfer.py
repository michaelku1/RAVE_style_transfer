#!/usr/bin/env python3
"""
Simple RAVE v2 Style Transfer Example

This script demonstrates the basic workflow for style transfer using RAVE v2 with AdaIN.
It shows the three main steps: learning target statistics, learning source statistics, and applying style transfer.

Usage:
    python simple_style_transfer.py --model /path/to/model --source source.wav --target target.wav --output output.wav
"""

import os
import sys
import argparse
import torch
import torchaudio

# Add current directory to path for rave import
sys.path.append(os.path.abspath('.'))

try:
    import rave
except ImportError:
    print("Error: Could not import rave. Make sure you're in the RAVE directory.")
    sys.exit(1)


def load_model(model_path, device='cuda'):
    """Load a RAVE model"""
    print(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    # Check if it's a scripted model (.ts file)
    if model_path.endswith('.ts'):
        model = torch.jit.load(model_path, map_location=device)
        print("Loaded scripted model")
        return model
    
    # Load regular model
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        raise ValueError(f"Config not found in {model_path}")
    
    import gin
    gin.parse_config_file(config_path)
    model = rave.RAVE()
    
    run = rave.core.search_for_run(model_path)
    if run is None:
        raise ValueError(f"Checkpoint not found in {model_path}")
    
    model = model.load_from_checkpoint(run)
    model = model.to(device)
    model.eval()
    print("Loaded regular model")
    
    return model


def load_audio(audio_path, target_sr=None, n_channels=None):
    """Load and preprocess audio file"""
    print(f"Loading audio from {audio_path}")
    
    try:
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        raise ValueError(f"Could not load audio file {audio_path}: {e}")
    
    # Resample if needed
    if target_sr and sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        print(f"Resampled from {sr}Hz to {target_sr}Hz")
    
    # Handle channel mismatch
    if n_channels and n_channels != audio.shape[0]:
        if n_channels < audio.shape[0]:
            audio = audio[:n_channels]
            print(f"Reduced channels from {audio.shape[0]} to {n_channels}")
        else:
            raise ValueError(f"Audio has {audio.shape[0]} channels but model expects {n_channels}")
    
    return audio


def check_adain_capability(model):
    """Check if the model supports AdaIN style transfer"""
    has_adain = False
    
    # Check if model has AdaIN flag
    if hasattr(model, 'is_using_adain'):
        has_adain = model.is_using_adain
    else:
        # Check if any module is AdaIN
        for module in model.modules():
            if hasattr(module, '__class__') and 'AdaptiveInstanceNormalization' in str(module.__class__):
                has_adain = True
                break
    
    if has_adain:
        print("✓ Model supports AdaIN style transfer")
    else:
        print("⚠ Model does not support AdaIN style transfer")
    
    return has_adain


def learn_statistics(model, audio_path, learn_target=True, num_chunks=5):
    """
    Learn statistics from audio file
    
    Args:
        model: RAVE model
        audio_path: Path to audio file
        learn_target: If True, learn target statistics; if False, learn source statistics
        num_chunks: Number of chunks to process for learning
    """
    print(f"Learning {'target' if learn_target else 'source'} statistics from {audio_path}")
    
    # Load audio
    audio = load_audio(audio_path, target_sr=model.sr, n_channels=model.n_channels)
    audio = audio.to(next(model.parameters()).device)
    
    # Set learning mode
    if hasattr(model, 'set_learn_target'):
        model.set_learn_target(learn_target)
        model.set_learn_source(not learn_target)
    
    # Process audio to learn statistics
    with torch.no_grad():
        # For demonstration, we'll process the audio in chunks
        chunk_size = min(8192, audio.shape[-1] // num_chunks)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, audio.shape[-1])
            
            if start_idx >= audio.shape[-1]:
                break
                
            chunk = audio[..., start_idx:end_idx]
            
            # Pad if needed
            if chunk.shape[-1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[-1]))
            
            # Process chunk
            if hasattr(model, 'encode'):
                model.encode(chunk.unsqueeze(0))
            else:
                # For scripted models, just pass through
                model(chunk.unsqueeze(0))
    
    print(f"✓ {'Target' if learn_target else 'Source'} statistics learned")


def apply_style_transfer(model, source_path, output_path):
    """
    Apply style transfer to source audio
    
    Args:
        model: RAVE model with learned statistics
        source_path: Path to source audio file
        output_path: Path to save output audio
    """
    print(f"Applying style transfer to {source_path}")
    
    # Load source audio
    source_audio = load_audio(source_path, target_sr=model.sr, n_channels=model.n_channels)
    source_audio = source_audio.to(next(model.parameters()).device)
    
    # Disable learning, enable style transfer
    if hasattr(model, 'set_learn_source'):
        model.set_learn_source(False)
        model.set_learn_target(False)
    
    # Apply style transfer
    with torch.no_grad():
        output_audio = model(source_audio.unsqueeze(0))
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, output_audio.cpu(), sample_rate=model.sr)
    print(f"✓ Style transfer completed. Output saved to {output_path}")


def reset_statistics(model):
    """Reset learned statistics"""
    print("Resetting learned statistics")
    
    if hasattr(model, 'set_reset_source'):
        model.set_reset_source(True)
        model.set_reset_target(True)
    
    print("✓ Statistics reset")


def main():
    parser = argparse.ArgumentParser(description="Simple RAVE v2 Style Transfer")
    parser.add_argument("--model", required=True, help="Path to trained RAVE model")
    parser.add_argument("--source", required=True, help="Path to source audio file")
    parser.add_argument("--target", required=True, help="Path to target style audio file")
    parser.add_argument("--output", required=True, help="Path to output audio file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num-chunks", type=int, default=5, help="Number of chunks for learning statistics")
    parser.add_argument("--reset", action="store_true", help="Reset learned statistics")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_model(args.model, device)
        
        # Check AdaIN capability
        has_adain = check_adain_capability(model)
        
        # Reset statistics if requested
        if args.reset:
            reset_statistics(model)
            return
        
        if not has_adain:
            print("Warning: Model does not support AdaIN. Style transfer may not work properly.")
        
        # Step 1: Learn target statistics
        learn_statistics(model, args.target, learn_target=True, num_chunks=args.num_chunks)
        
        # Step 2: Learn source statistics
        learn_statistics(model, args.source, learn_target=False, num_chunks=args.num_chunks)
        
        # Step 3: Apply style transfer
        apply_style_transfer(model, args.source, args.output)
        
        print("\n🎵 Style transfer workflow completed successfully!")
        print(f"Source: {args.source}")
        print(f"Target style: {args.target}")
        print(f"Output: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 