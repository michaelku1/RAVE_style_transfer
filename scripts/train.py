import hashlib
import os
import sys
import time
from typing import Any, Dict

import gin
import pytorch_lightning as pl
import torch
from absl import flags, app
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import GPUtil

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

import rave
import rave.core
import rave.dataset
from rave.transforms import get_augmentations, add_augmentation


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_multi_string('augment',
                           default = [],
                            help = 'augmentation configurations to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_string('out_path',
                    default="runs/",
                    help='Output folder')
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     500000,
                     help='save every n steps (default: just last)')
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('channels', 0, help="number of audio channels")
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
flags.DEFINE_list('rand_pitch',
                  default=None,
                  help='activates random pitch')
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
flags.DEFINE_bool('smoke_test', 
                  default=False,
                  help="Run training with n_batches=1 to test the model")
# Profiling flags
flags.DEFINE_bool('enable_profiling',
                  default=False,
                  help='Enable PyTorch profiler for detailed performance analysis')
flags.DEFINE_bool('enable_memory_profiling',
                  default=False,
                  help='Enable memory usage profiling')
flags.DEFINE_bool('enable_timing_profiling',
                  default=False,
                  help='Enable detailed timing profiling')
flags.DEFINE_integer('profile_steps',
                     default=100,
                     help='Number of steps to profile when profiling is enabled')


class EMA(pl.Callback):

    def __init__(self, factor=.999) -> None:
        super().__init__()
        self.weights = {}
        self.factor = factor

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx) -> None:
        for n, p in pl_module.named_parameters():
            if n not in self.weights:
                self.weights[n] = p.data.clone()
                continue

            self.weights[n] = self.weights[n] * self.factor + p.data * (
                1 - self.factor)

    def swap_weights(self, module):
        for n, p in module.named_parameters():
            current = p.data.clone()
            p.data.copy_(self.weights[n])
            self.weights[n] = current

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def state_dict(self) -> Dict[str, Any]:
        return self.weights.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.weights.update(state_dict)


class MemoryProfiler(pl.Callback):
    """Callback to profile memory usage during training."""
    
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.memory_log = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            # CPU memory
            cpu_memory = psutil.virtual_memory().percent
            
            # GPU memory
            gpu_memory = 0
            gpu_memory_reserved = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            memory_info = {
                'step': trainer.global_step,
                'cpu_memory_percent': cpu_memory,
                'gpu_memory_allocated_gb': gpu_memory,
                'gpu_memory_reserved_gb': gpu_memory_reserved,
                'batch_idx': batch_idx
            }
            
            self.memory_log.append(memory_info)
            
            # Log to console
            print(f"Step {trainer.global_step}: CPU: {cpu_memory:.1f}%, "
                  f"GPU: {gpu_memory:.2f}GB allocated")
    
    def on_train_end(self, trainer, pl_module):
        # Save memory log to file
        import json
        log_path = os.path.join(trainer.logger.log_dir, 'memory_profile.json')
        with open(log_path, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
        print(f"Memory profile saved to {log_path}")


class TimingProfiler(pl.Callback):
    """Callback to profile timing of different training components."""
    
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.timing_log = []
        self.batch_start_time = None
        self.forward_start_time = None
        self.backward_start_time = None
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            self.batch_start_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0 and self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            
            timing_info = {
                'step': trainer.global_step,
                'batch_idx': batch_idx,
                'total_batch_time': batch_time,
                'samples_per_second': batch[0].shape[0] / batch_time if batch[0].shape[0] > 0 else 0
            }
            
            self.timing_log.append(timing_info)
            
            # Log to console
            print(f"Step {trainer.global_step}: Batch time: {batch_time:.3f}s, "
                  f"Samples/sec: {timing_info['samples_per_second']:.1f}")
    
    def on_train_end(self, trainer, pl_module):
        # Save timing log to file
        import json
        log_path = os.path.join(trainer.logger.log_dir, 'timing_profile.json')
        with open(log_path, 'w') as f:
            json.dump(self.timing_log, f, indent=2)
        print(f"Timing profile saved to {log_path}")


class DetailedProfiler(pl.Callback):
    """Callback to run detailed PyTorch profiler for specific steps."""
    
    def __init__(self, profile_steps=100, output_dir="profiler_output"):
        super().__init__()
        self.profile_steps = profile_steps
        self.output_dir = output_dir
        self.profiler = None
        self.profiling_active = False
        
    def on_train_start(self, trainer, pl_module):
        if FLAGS.enable_profiling:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Detailed profiling enabled. Will profile {self.profile_steps} steps.")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (FLAGS.enable_profiling and 
            trainer.global_step >= 10 and  # Skip first few steps for warmup
            trainer.global_step < 10 + self.profile_steps and
            not self.profiling_active):
            
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=5,
                    active=self.profile_steps,
                    repeat=1
                )
            )
            self.profiler.start()
            self.profiling_active = True
            print(f"Starting detailed profiling at step {trainer.global_step}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.profiling_active and self.profiler is not None:
            self.profiler.step()
            
            if trainer.global_step >= 10 + self.profile_steps:
                self.profiler.stop()
                self.profiling_active = False
                
                # Save profiler results
                profiler_output = os.path.join(self.output_dir, f"profile_step_{trainer.global_step}")
                self.profiler.export_chrome_trace(f"{profiler_output}.json")
                self.profiler.export_stacks(f"{profiler_output}_stacks.txt")
                
                print(f"Profiling completed. Results saved to {profiler_output}")
                
                # Print summary
                print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name

def parse_augmentations(augmentations):
    for a in augmentations:
        gin.parse_config_file(a)
        add_augmentation()
        gin.clear_config()
    return get_augmentations()

def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # check dataset channels
    n_channels = rave.dataset.get_training_channels(FLAGS.db_path, FLAGS.channels)
    gin.bind_parameter('RAVE.n_channels', n_channels)

    # parse augmentations
    augmentations = parse_augmentations(map(add_gin_extension, FLAGS.augment))
    gin.bind_parameter('dataset.get_dataset.augmentations', augmentations)

    # parse configuration
    if FLAGS.ckpt:
        config_file = rave.core.search_for_config(FLAGS.ckpt)
        if config_file is None:
            print('Config file not found in %s'%FLAGS.run)
        gin.parse_config_file(config_file)
        config_names = [os.path.basename(config_file)]
    else:
        config_files = list(map(add_gin_extension, FLAGS.config))
        gin.parse_config_files_and_bindings(
            config_files,
            FLAGS.override,
        )
        config_names = [os.path.basename(cfg) for cfg in config_files]

    # Save config names to a file in the output directory
    config_name_path = os.path.join(FLAGS.out_path, FLAGS.name + '_config_name.txt')
    with open(config_name_path, 'w') as f:
        f.write('\n'.join(config_names))

    # create model
    model = rave.RAVE(n_channels=FLAGS.channels)
    # Optionally, store config names in the model for later reference
    model.config_names = config_names
    if FLAGS.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    # parse datasset
    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       model.sr,
                                       FLAGS.n_signal,
                                       derivative=FLAGS.derivative,
                                       normalize=FLAGS.normalize,
                                       rand_pitch=FLAGS.rand_pitch,
                                       n_channels=n_channels)
    train, val = rave.dataset.split_dataset(dataset, 98)

    # get data-loader
    num_workers = FLAGS.workers
    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0
    train = DataLoader(train,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_filename = "last" if FLAGS.save_every is None else "epoch-{epoch:04d}"                                                        
    last_checkpoint = rave.core.ModelCheckpoint(filename=last_filename, step_period=FLAGS.save_every)

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = 1 if FLAGS.smoke_test else FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    if FLAGS.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]

    RUN_NAME = f'{FLAGS.name}_{gin_hash}'

    os.makedirs(os.path.join(FLAGS.out_path, RUN_NAME), exist_ok=True)

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
        rave.model.WarmupCallback(),
        rave.model.QuantizeCallback(),
        # rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
        rave.model.BetaWarmupCallback(),
    ]

    # Add profiling callbacks
    if FLAGS.enable_memory_profiling:
        callbacks.append(MemoryProfiler(log_every_n_steps=100))
        print("Memory profiling enabled")
    
    if FLAGS.enable_timing_profiling:
        callbacks.append(TimingProfiler(log_every_n_steps=100))
        print("Timing profiling enabled")
    
    if FLAGS.enable_profiling:
        callbacks.append(DetailedProfiler(profile_steps=FLAGS.profile_steps))
        print("Detailed PyTorch profiling enabled")

    if FLAGS.ema is not None:
        callbacks.append(EMA(FLAGS.ema))

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            FLAGS.out_path,
            name=RUN_NAME,
        ),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=FLAGS.max_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        print('loading state from file %s'%run)
        loaded = torch.load(run, map_location='cpu')
        # model = model.load_state_dict(loaded)
        trainer.fit_loop.epoch_loop._batches_that_stepped = loaded['global_step']
        # model = model.load_state_dict(loaded['state_dict'])
    
    with open(os.path.join(FLAGS.out_path, RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    # Start training with timing
    print(f"Starting training for {FLAGS.max_steps} steps...")
    training_start_time = time.time()
    
    trainer.fit(model, train, val, ckpt_path=run)
    
    # Calculate and log total training time
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETED")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Average time per step: {total_training_time/FLAGS.max_steps:.3f}s")
    print(f"Steps per second: {FLAGS.max_steps/total_training_time:.2f}")
    print(f"{'='*50}")
    
    # Save training summary
    training_summary = {
        'total_training_time_seconds': total_training_time,
        'total_steps': FLAGS.max_steps,
        'average_time_per_step': total_training_time/FLAGS.max_steps,
        'steps_per_second': FLAGS.max_steps/total_training_time,
        'training_start_time': training_start_time,
        'training_end_time': time.time()
    }
    
    import json
    summary_path = os.path.join(trainer.logger.log_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training summary saved to {summary_path}")


if __name__ == "__main__": 
    app.run(main)
