#!/usr/bin/env python3
"""
RAVE v2 Style Transfer Inference Script

This script demonstrates how to perform style transfer using RAVE v2 with AdaIN.
It supports both batch processing and real-time streaming modes.

Usage:
    python style_transfer.py --model /path/to/model --source /path/to/source.wav --target /path/to/target.wav --output /path/to/output.wav
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cached_conv as cc

try:
    import rave
except ImportError:
    sys.path.append(os.path.abspath('.'))
    import rave


class StyleTransferInference:
    """RAVE v2 Style Transfer Inference Class"""
    
    def __init__(self, model_path, device='cuda', streaming=False):
        """
        Initialize the style transfer model
        
        Args:
            model_path: Path to the trained RAVE model
            device: Device to run inference on ('cuda' or 'cpu')
            streaming: Whether to use streaming mode for real-time processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.streaming = streaming
        self.model = self._load_model(model_path)
        self.receptive_field = rave.core.get_minimum_size(self.model)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model sampling rate: {self.model.sr}")
        print(f"Model channels: {self.model.n_channels}")
        print(f"Receptive field: {self.receptive_field}")
        print(f"Using AdaIN: {self._has_adain()}")
        
    def _load_model(self, model_path):
        """Load the RAVE model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
            
        # Check if it's a scripted model (.ts file)
        if model_path.endswith('.ts'):
            model = torch.jit.load(model_path, map_location=self.device)
            print("Loaded scripted model")
        else:
            # Load regular model
            config_path = rave.core.search_for_config(model_path)

            breakpoint()
            if config_path is None:
                raise ValueError(f"Config not found in {model_path}")
            
            import gin
            gin.parse_config_file(config_path)
            model = rave.RAVE()
            
            run = rave.core.search_for_run(model_path)
            if run is None:
                raise ValueError(f"Checkpoint not found in {model_path}")
                
            model = model.load_from_checkpoint(run)
            model = model.to(self.device)
            model.eval()
            print("Loaded regular model")
            
        return model
    
    def _has_adain(self):
        """Check if the model uses AdaIN"""
        if hasattr(self.model, 'is_using_adain'):
            return self.model.is_using_adain
        else:
            # Check if any module is AdaIN
            for module in self.model.modules():
                if isinstance(module, rave.blocks.AdaptiveInstanceNormalization):
                    return True
            return False
    
    def _load_audio(self, audio_path, target_sr=None):
        """Load and preprocess audio file"""
        try:
            audio, sr = torchaudio.load(audio_path)
        except Exception as e:
            raise ValueError(f"Could not load audio file {audio_path}: {e}")
        
        # Resample if needed
        if target_sr and sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            print(f"Resampled from {sr}Hz to {target_sr}Hz")
        
        # Handle channel mismatch
        if self.model.n_channels != audio.shape[0]:
            if self.model.n_channels < audio.shape[0]:
                audio = audio[:self.model.n_channels]
                print(f"Reduced channels from {audio.shape[0]} to {self.model.n_channels}")
            else:
                raise ValueError(f"Audio has {audio.shape[0]} channels but model expects {self.model.n_channels}")
        
        return audio.to(self.device)
    
    def _prepare_chunks(self, audio, chunk_size=None):
        """Prepare audio chunks for streaming processing"""
        if not self.streaming:
            return [audio]
        
        if chunk_size is None:
            chunk_size = max(self.receptive_field * 2, 8192)  # Default chunk size
        
        if chunk_size <= self.receptive_field:
            raise ValueError(f"Chunk size {chunk_size} must be larger than receptive field {self.receptive_field}")
        
        # Split audio into chunks
        chunks = []
        for i in range(0, audio.shape[-1], chunk_size):
            chunk = audio[..., i:i + chunk_size]
            if chunk.shape[-1] < chunk_size:
                # Pad last chunk
                pad_size = chunk_size - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            chunks.append(chunk)
        
        return chunks
    
    def learn_source_statistics(self, source_audio_path, num_chunks=10):
        """
        Learn source audio statistics for style transfer
        
        Args:
            source_audio_path: Path to source audio file
            num_chunks: Number of chunks to use for learning statistics
        """
        if not self._has_adain():
            print("Warning: Model does not use AdaIN. Style transfer may not work properly.")
            return
        
        print(f"Learning source statistics from {source_audio_path}")
        
        # Load source audio
        source_audio = self._load_audio(source_audio_path, self.model.sr)
        chunks = self._prepare_chunks(source_audio)
        
        # NOTE 重點 Enable source learning
        if hasattr(self.model, 'set_learn_source'):
            self.model.set_learn_source(True)
            self.model.set_learn_target(False)
        
        # Process chunks to learn statistics
        with torch.no_grad():
            for i, chunk in enumerate(tqdm(chunks[:num_chunks], desc="Learning source statistics")):
                if self.streaming:
                    self.model.encode(chunk.unsqueeze(0))
                else:
                    self.model.encode(source_audio.unsqueeze(0))
                    break  # Only need one pass for non-streaming
        
        print("Source statistics learned successfully")
    
    def learn_target_statistics(self, target_audio_path, num_chunks=10):
        """
        Learn target style statistics for style transfer
        
        Args:
            target_audio_path: Path to target style audio file
            num_chunks: Number of chunks to use for learning statistics
        """
        if not self._has_adain():
            print("Warning: Model does not use AdaIN. Style transfer may not work properly.")
            return
        
        print(f"Learning target statistics from {target_audio_path}")
        
        # Load target audio
        target_audio = self._load_audio(target_audio_path, self.model.sr)
        chunks = self._prepare_chunks(target_audio)
        
        # Enable target learning
        if hasattr(self.model, 'set_learn_target'):
            self.model.set_learn_target(True)
            self.model.set_learn_source(False)
        
        # Process chunks to learn statistics
        with torch.no_grad():
            for i, chunk in enumerate(tqdm(chunks[:num_chunks], desc="Learning target statistics")):
                if self.streaming:
                    self.model.encode(chunk.unsqueeze(0))
                else:
                    self.model.encode(target_audio.unsqueeze(0))
                    break  # Only need one pass for non-streaming
        
        print("Target statistics learned successfully")
    
    def apply_style_transfer(self, source_audio_path, output_path, chunk_size=None):
        """
        Apply style transfer to source audio
        
        Args:
            source_audio_path: Path to source audio file
            output_path: Path to save the output audio
            chunk_size: Chunk size for streaming processing
        """
        if not self._has_adain():
            print("Warning: Model does not use AdaIN. Style transfer may not work properly.")
        
        print(f"Applying style transfer to {source_audio_path}")
        
        # Load source audio
        source_audio = self._load_audio(source_audio_path, self.model.sr)
        chunks = self._prepare_chunks(source_audio, chunk_size)
        
        # Disable learning, enable style transfer
        if hasattr(self.model, 'set_learn_source'):
            self.model.set_learn_source(False)
            self.model.set_learn_target(False)
        
        # Process audio with style transfer
        output_chunks = []
        with torch.no_grad():
            for chunk in tqdm(chunks, desc="Applying style transfer"):
                if self.streaming:
                    output_chunk = self.model(chunk.unsqueeze(0))
                else:
                    output_chunk = self.model(source_audio.unsqueeze(0))
                    output_chunks = [output_chunk]
                    break
                
                output_chunks.append(output_chunk)
        
        # Concatenate chunks and save
        if len(output_chunks) > 1:
            output_audio = torch.cat(output_chunks, dim=-1)
        else:
            output_audio = output_chunks[0]
        
        # Remove padding if it was added
        if self.streaming and output_audio.shape[-1] > source_audio.shape[-1]:
            output_audio = output_audio[..., :source_audio.shape[-1]]
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, output_audio.cpu(), sample_rate=self.model.sr)
        print(f"Style transfer completed. Output saved to {output_path}")
    
    def reset_statistics(self):
        """Reset learned statistics"""
        if hasattr(self.model, 'set_reset_source'):
            self.model.set_reset_source(True)
            self.model.set_reset_target(True)
        print("Statistics reset")
    
    def batch_style_transfer(self, source_dir, target_audio_path, output_dir, num_chunks=10):
        """
        Apply style transfer to multiple source files
        
        Args:
            source_dir: Directory containing source audio files
            target_audio_path: Path to target style audio file
            output_dir: Directory to save output files
            num_chunks: Number of chunks to use for learning statistics
        """
        # Learn target statistics once
        self.learn_target_statistics(target_audio_path, num_chunks)
        
        # Get all audio files in source directory
        source_files = []
        valid_exts = rave.core.get_valid_extensions()
        for ext in valid_exts:
            source_files.extend(Path(source_dir).glob(f"**/*{ext}"))
        
        print(f"Found {len(source_files)} source files")
        
        # Process each source file
        for source_file in tqdm(source_files, desc="Processing source files"):
            # Learn source statistics for this file
            self.learn_source_statistics(str(source_file), num_chunks)
            
            # Generate output path
            relative_path = source_file.relative_to(source_dir)
            output_path = Path(output_dir) / relative_path
            
            # Apply style transfer
            self.apply_style_transfer(str(source_file), str(output_path))
            
            # Reset source statistics for next file
            self.reset_statistics()


def main():
    parser = argparse.ArgumentParser(description="RAVE v2 Style Transfer")
    parser.add_argument("--model", required=True, help="Path to trained RAVE model")
    parser.add_argument("--source", help="Path to source audio file or directory")
    parser.add_argument("--target", help="Path to target style audio file")
    parser.add_argument("--output", help="Path to output audio file or directory")
    # parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device to use, e.g., cuda:0 or cpu')
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--chunk-size", type=int, help="Chunk size for streaming mode")
    parser.add_argument("--num-chunks", type=int, default=10, help="Number of chunks for learning statistics")
    parser.add_argument("--batch", action="store_true", help="Process multiple source files")
    parser.add_argument("--reset", action="store_true", help="Reset learned statistics")
    
    args = parser.parse_args()
    
    # Initialize style transfer
    style_transfer = StyleTransferInference(
        model_path=args.model,
        device=args.device,
        streaming=args.streaming
    )
    
    # Reset statistics if requested
    if args.reset:
        style_transfer.reset_statistics()
        return
    
    # Batch processing
    if args.batch:
        if not args.source or not args.target or not args.output:
            print("Error: --batch mode requires --source, --target, and --output")
            return
        
        style_transfer.batch_style_transfer(
            source_dir=args.source,
            target_audio_path=args.target,
            output_dir=args.output,
            num_chunks=args.num_chunks
        )
    
    # Single file processing
    else:
        if not args.source or not args.target or not args.output:
            print("Error: Single file mode requires --source, --target, and --output")
            return
        
        # Learn target statistics
        style_transfer.learn_target_statistics(args.target, args.num_chunks)
        
        # Learn source statistics
        style_transfer.learn_source_statistics(args.source, args.num_chunks)
        
        # Apply style transfer
        style_transfer.apply_style_transfer(args.source, args.output, args.chunk_size)


if __name__ == "__main__":
    main() 