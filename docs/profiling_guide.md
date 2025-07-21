# RAVE v3 Training Profiling Guide

This guide explains how to use the profiling features added to the RAVE v3 training script to analyze training performance and identify bottlenecks.

## Overview

The profiling system provides three main types of profiling:

1. **Timing Profiling** - Measures batch processing times and throughput
2. **Memory Profiling** - Tracks CPU and GPU memory usage
3. **Detailed PyTorch Profiling** - Provides detailed GPU/CPU operation analysis

## Quick Start

### Basic Profiling (Timing + Memory)

```bash
python scripts/train.py \
    --name rave_v3_profiled \
    --config v3.gin \
    --db_path /path/to/your/dataset \
    --enable_timing_profiling True \
    --enable_memory_profiling True \
    --max_steps 1000 \
    --batch 4
```

### Detailed PyTorch Profiling

```bash
python scripts/train.py \
    --name rave_v3_detailed_profiled \
    --config v3.gin \
    --db_path /path/to/your/dataset \
    --enable_profiling True \
    --profile_steps 100 \
    --max_steps 2000 \
    --batch 4
```

### Full Profiling (All Features)

```bash
python scripts/train.py \
    --name rave_v3_full_profiled \
    --config v3.gin \
    --db_path /path/to/your/dataset \
    --enable_timing_profiling True \
    --enable_memory_profiling True \
    --enable_profiling True \
    --profile_steps 100 \
    --max_steps 2000 \
    --batch 4
```

## Profiling Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable_profiling` | bool | False | Enable detailed PyTorch profiler |
| `--enable_memory_profiling` | bool | False | Enable memory usage profiling |
| `--enable_timing_profiling` | bool | False | Enable timing profiling |
| `--profile_steps` | int | 100 | Number of steps to profile with PyTorch profiler |

## Profiling Outputs

### 1. Timing Profile (`timing_profile.json`)

Contains step-by-step timing information:

```json
[
  {
    "step": 100,
    "batch_idx": 100,
    "total_batch_time": 0.045,
    "samples_per_second": 177.78
  }
]
```

**Metrics:**
- `total_batch_time`: Time to process one batch (seconds)
- `samples_per_second`: Audio samples processed per second

### 2. Memory Profile (`memory_profile.json`)

Tracks memory usage throughout training:

```json
[
  {
    "step": 100,
    "cpu_memory_percent": 45.2,
    "gpu_memory_allocated_gb": 2.34,
    "gpu_memory_reserved_gb": 2.45,
    "batch_idx": 100
  }
]
```

**Metrics:**
- `cpu_memory_percent`: System RAM usage percentage
- `gpu_memory_allocated_gb`: GPU memory actually used (GB)
- `gpu_memory_reserved_gb`: GPU memory reserved by PyTorch (GB)

### 3. Detailed Profiler Outputs

When using `--enable_profiling True`:

- **Chrome Trace Files** (`profile_step_*.json`): Can be opened in Chrome's `chrome://tracing/`
- **Stack Traces** (`profile_step_*_stacks.txt`): Detailed function call stacks
- **Console Summary**: Top 10 most time-consuming operations

### 4. Training Summary (`training_summary.json`)

Overall training performance metrics:

```json
{
  "total_training_time_seconds": 3600.0,
  "total_steps": 1000,
  "average_time_per_step": 3.6,
  "steps_per_second": 0.278,
  "training_start_time": 1640995200.0,
  "training_end_time": 1640998800.0
}
```

## Analyzing Profiling Data

### Performance Bottlenecks

1. **High GPU Memory Usage**: Consider reducing batch size
2. **Low Samples/Second**: Check for CPU bottlenecks or inefficient operations
3. **Long Batch Times**: Look for expensive operations in PyTorch profiler output

### Memory Optimization

- Monitor `gpu_memory_reserved_gb` vs `gpu_memory_allocated_gb`
- Large difference indicates memory fragmentation
- Consider using gradient checkpointing for large models

### Throughput Optimization

- Target: >100 samples/second for real-time applications
- Monitor `samples_per_second` metric
- Optimize data loading if CPU memory usage is high

## Using Chrome Trace Viewer

1. Open Chrome and navigate to `chrome://tracing/`
2. Load the `profile_step_*.json` file
3. Analyze:
   - GPU operations (green bars)
   - CPU operations (blue bars)
   - Memory operations (red bars)
   - Data transfer operations (orange bars)

## Example Analysis Workflow

1. **Start with basic profiling** to get overview:
   ```bash
   --enable_timing_profiling True --enable_memory_profiling True
   ```

2. **If performance issues detected**, run detailed profiling:
   ```bash
   --enable_profiling True --profile_steps 100
   ```

3. **Analyze outputs**:
   - Check `timing_profile.json` for throughput issues
   - Check `memory_profile.json` for memory bottlenecks
   - Use Chrome trace viewer for detailed operation analysis

4. **Optimize based on findings**:
   - Adjust batch size
   - Modify model architecture
   - Optimize data loading
   - Use mixed precision training

## Troubleshooting

### Common Issues

1. **Profiling slows down training**: Reduce `--profile_steps` or disable detailed profiling
2. **Memory errors during profiling**: Reduce batch size or enable gradient checkpointing
3. **Chrome trace files too large**: Reduce `--profile_steps` or use sampling

### Performance Tips

- Profile on representative data (not just first few batches)
- Use smaller batch sizes for profiling to avoid memory issues
- Run profiling after model has warmed up (skip first 10-50 steps)
- Compare profiling results across different hardware configurations

## Integration with TensorBoard

Profiling data is automatically saved to the TensorBoard log directory. You can view some metrics in TensorBoard:

```bash
tensorboard --logdir runs/
```

## Example Script

Use the provided example script for easy profiling:

```bash
python examples/profiling_example.py
```

This interactive script provides pre-configured profiling commands and explains the outputs. 