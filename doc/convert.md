# ExLlamaV2

### Arguments

Here are the arguments to `convert.py`:

- **-i / --in_dir *directory***: _(required if not resuming)_ The source model to convert, in HF format (FP16). The directory should 
contain at least a `config.json` file, a `tokenizer.model` file and one or more `.safetensors` files containing weights.
If there are multiple weights files, they will all be indexed and searched for the neccessary tensors, so sharded models are 
supported.
  

- **-o / --out_dir *directory***: _(required)_ A working directory where the converter can store temporary files and deposit
the final output unless **-cf** is also provided. If this directory is not empty when the conversion starts, the converter
will attempt to resume whatever was being worked on in that directory. So if a job is interrupted, you can rerun the
script with the same arguments to pick up from where it stopped. Note that parameters are read from the `job.json`
file kept in this directory, so you won't be able to supply new parameters to a resumed job without editing that file.
If you do, note that not all changes would leave the job in a valid state, so be careful with that.
  
  
- **-nr / --no_resume**: If this flag is specified, the converter will not resume an interrupted job even if you point it
to a non-empty working directory. If the working directory is not empty, all files within it will be deleted and the
converter will start a new job.
  
  
- **-cf / --compile_full *directory***: By default the resulting `.safetensors` files are saved to the working directory.
If you specify a directory with **-cf**, the quantized weights will be saved there instead. In addition, all files from
the model (input) directory will be copied to this directory except for the original `.safetensors` files, resulting in
a full model directory that can be used for inference by ExLlamaV2.
  

- **-om / --output_measurement *file***: After the first (measurement) pass is completed, a `measurement.json` file will
be dropped in the working directory (**-o**). If you specify **-om** with a path, the measurement will be saved to that
path after the measurement pass, and the script will exit immediately after.
  

- **-m / --measurement *file***: Skip the measurement pass and instead use the results from the provided file. This is
particularly useful when quantizing the same model to multiple bitrates, since the measurement pass can take a long time
to complete.
  

- **-c / --cal_dataset *file***: (_optional_) The calibration dataset in Parquet format. The quantizer concatenates all
the data in this file into one long string and uses the first _r_ \* _l_ tokens for calibration. If this is not
specified, the default, built-in calibration dataset is used which contains a broad mix of different types of data. It's
designed to prevent the quantized model from overfitting to any particular mode, language or style, and generally
results in more robust, reliable outputs, especially at lower bitrates.
  

- **-l / --length *int***: Length, in tokens, of each calibration row. Default is 2048.
  

- **-r / --dataset_rows *int***: Number of rows in the calibration batch. Default is 100.
  

- **-ml / --measurement_length *int***: Length, in tokens, of each calibration row used for the measuring pass. Default
is 2048. 
  

- **-mr / --measurement_rows *int***: Number of rows in the calibration batch for the measuring pass. Default is 16.
  

- **-b / --bits *float***: Target average number of bits per weight.
  

- **-hb / --bits *int***: Number of bits for the lm_head (output) layer of the model. Default is 6, although that
value actually results in a mixed-precision quantization of about 6.3 bits. Options are 2, 3, 4, 5, 6 and 8. (Only 6
and 8 appear to be useful.)

  
- **-ss / --shard_size *float***: Output shard size, in megabytes. Default is 8192. Set this to 0 to disable sharding.
Note that writing a very large `.safetensors` file can require a lot of system RAM.


- **-ra / --rope_alpha *float***: RoPE (NTK) alpha to apply to base model for calibration.


- **-rs / --rope_scale *float***: RoPE scaling factor to apply to base model for calibration. This settings is not 
automatically read from the model's config, so it's strongly recommended that you check what setting the model was
trained/finetuned with. E.g.: deepseek-coder uses a scaling factor of 4, so will be incorrectly calibrated if you
convert it without `-rs 4`.


### Notes

The converter works in two passes; first it measures how quantization impacts each module of the model, and then it
actually quantizes the model, choosing quantization parameters for each layer that minimize the overall error while 
also achieving the desired overall (average) bitrate.

The first pass is slow, since it effectively quantizes the entire model about 12 times over (albeit with a less
comprehensive sample of the calibration dataset), so make sure to save the `measurement.json` file so you can skip the
measurement pass on subsequent quants of the same model.

### Examples

Convert a model and create a directory containing the quantized version with all of its original files:

```sh
python convert.py \
    -i /mnt/models/llama2-7b-fp16/ \
    -o /mnt/temp/exl2/ \
    -cf /mnt/models/llama2-7b-exl2/3.0bpw/ \
    -b 3.0 
```

Run just the measurement pass on a model, clearing the working directory first:

```sh
python convert.py \
    -i /mnt/models/llama2-7b-fp16/ \
    -o /mnt/temp/exl2/ \
    -nr \
    -om /mnt/models/llama2-7b-exl2/measurement.json
```

Use that measurement to quantize the model at two different bitrates:

```sh
python convert.py \
    -i /mnt/models/llama2-7b-fp16/ \
    -o /mnt/temp/exl2/ \
    -nr \
    -m /mnt/models/llama2-7b-exl2/measurement.json \
    -cf /mnt/models/llama2-7b-exl2/4.0bpw/ \
    -b 4.0
    
python convert.py \
    -i /mnt/models/llama2-7b-fp16/ \
    -o /mnt/temp/exl2/ \
    -nr \
    -m /mnt/models/llama2-7b-exl2/measurement.json \
    -cf /mnt/models/llama2-7b-exl2/4.5bpw/ \
    -b 4.5
```

If the working `-o` directory is not empty and you do not specify `-nr`, any existing job in that directory
will be resumed. You can resume a job with no other arguments:

```sh
python convert.py -o /mnt/temp/exl2/
```

### Notes

- If the conversion script seems to stop on the "Solving..." step, give it a moment. It's attempting to find the 
combination of quantization parameters within the bits budget that minimizes the product of measured errors per
individual layer, and the implementation is not very efficient.
- During measurement and conversion of MoE models you may see a message like: 
`!! Warning: w2.7 has less than 10% calibration for 77/115 rows`. This happens when a particular expert isn't triggered
enough during the reference forward passes to get a good amount of calibration data. It won't cause the
conversion to fail, and it may not be a big deal at all, but GPTQ-style quantization of MoE models is very new so I'm
not yet sure if it actually matters.
- After conversion, the "calibration perplexity (quant)" is a perplexity calculation on a small sample of the 
calibration data as processed by the quantized model under construction. If it looks too high (30 or more), 
quantization likely didn't go well, and if it's unreasonably high (in the thousands, for instance) quantization failed
catastrophically. 

### Hardware requirements

Roughly speaking, you'll need about 64 GB of RAM and 24 GB of VRAM to convert a 70B model, while 7B seems to require
about 16 GB of RAM and about 8 GB of VRAM.

The deciding factor for the memory requirement is the *width* of the model rather than the depth, so 120B models that
have the same hidden size as 70B models have the same hardware requirements. Mixtral 8x7B has much wider feed-forward
layers so requires about 20 GB of VRAM.

---

## Multi-GPU quantization (ReadyArt fork)

This fork can spread the conversion/quantization passes across several GPUs. Two mechanisms:

1. **Device distribution (`-d`)** — independent tensors are placed on different devices during the
   measurement and quantization passes. Tensors are *never split*; whole tensors are distributed.
2. **Calibration-row sharding (`-crs on`)** — the calibration batch is split across devices, with one
   module replica per device, so the (otherwise single-device) measurement pass runs in parallel.

### Arguments

- **-d / --devices *list***: Comma-separated CUDA device indices to use, e.g. `0,1`. Default: all
  visible devices. **The first entry is the primary** and owns every Hessian inversion; the remaining
  devices take independent tensors. Example: `-d 0,1` uses GPUs 0 and 1, with GPU 0 primary.

- **-dr / --device_ratios *list***: Comma-separated relative throughput weights, one per device in
  `-d`, e.g. `1,2` if the second card is about twice as fast. Default: learned from the measurement
  pass (seeded by an architecture prior). Use this to balance work across heterogeneous GPUs.

- **-crs / --calib_row_shards *on|off***: Shard calibration rows across the `-d` devices, one module
  replica per device. **Off by default**: it needs enough VRAM for a replica on *each* device, and any
  device that cannot hold one is dropped from the shard. Turn it on (`-crs on`) when each GPU has
  headroom — it parallelizes the measurement pass, which is otherwise single-device and leaves the
  other GPUs idle for the first half of the run.

- **-ps / --pinned_staging *on|off***: Stage weights to the host through pinned memory on a side stream
  instead of a synchronous pageable copy (default `on`). A single-GPU win; produces bit-identical output.

- **-hb / --head_bits *int***: Bits for the `lm_head` (output) layer. Use **`16`** to leave the head
  unquantized (fp16) — recommended for Gemma-4 (large 262k vocabulary). Default 6.

### Example: Gemma-4-31B on two GPUs

```sh
python convert.py \
    -i /mnt/models/gemma-4-31B-it \
    -o /mnt/models/gemma-4-31b-exl2-6.0bpw-hb16 \
    -cf /mnt/models/gemma-4-31b-exl2-6.0bpw-hb16 \
    -b 6.0 -hb 16 \
    -d 0,1 -crs on \
    -l 2048 -r 100
```

A ready-made wrapper for this exact job is in [`scripts/remote_convert_2gpu_hb16.sh`](../scripts/remote_convert_2gpu_hb16.sh).

Notes:
- Without `-crs on`, the measurement pass is single-device (the primary); `-d` still distributes the
  quantization of independent tensors. With `-crs on`, both passes use all listed devices.
- The primary device (first in `-d`) does every Hessian inversion, so give it the most VRAM headroom.
- Multi-GPU quantization changes *where* work runs, not the result: output is equivalent to the
  single-device path for the same bitrate/calibration.

## Running the converted model on multiple GPUs

After conversion, run the EXL2 model tensor-parallel across GPUs (this is an **inference** feature,
separate from the multi-GPU quantization above):

- **Single-process TP** — `test_inference.py --gpu_split auto`, or in code:
  ```python
  model.load_tp([20, 20, 20, 20],                    # VRAM (GiB) per GPU
                expect_cache_tokens=max_seq_len,
                expect_cache_base=ExLlamaV2Cache_SWA)
  cache = ExLlamaV2Cache_TP(model, base=ExLlamaV2Cache_SWA, max_seq_len=max_seq_len)
  ```
- **Multiprocess backend (MP)** — `exllamav2.tp_mp.TPProcessGroup` runs one worker process per GPU for
  higher decode throughput. See the [README](../README.md) and [`mtp/`](../mtp/) benchmarks.
- **MTP speculative decoding** works on both TP and MP via the Gemma-4 assistant drafter
  (`ExLlamaV2StreamingGenerator(..., mtp_drafter=...)` or `MTPMPDriver`).

Requirements for the TP/MP path:
- **FlashInfer ≥ 0.7.4** is mandatory (`pip install "flashinfer-python>=0.7.4"` then
  `flashinfer download-kernels`). It provides head_dim 512 decode on sm86 (RTX 3090).
- For Gemma-4, use the **SWA ring cache** (`ExLlamaV2Cache_SWA` base) so sliding-layer K/V stays
  bounded at long context; the TP cache wrapper is `ExLlamaV2Cache_TP`.
- TP CUDA graphs are on by default (`EXL2_TP_GRAPHS=1`); set `EXL2_TP_GRAPHS=0` to disable for debugging.  
