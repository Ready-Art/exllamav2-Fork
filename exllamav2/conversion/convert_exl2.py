from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.architecture import RopeStyle
import argparse, os, shutil
import sys
import json
from exllamav2.conversion.tokenize import tokenize
from exllamav2.conversion.measure import embeddings, measure_quant
from exllamav2.conversion.quantize import quant
from exllamav2.conversion.optimize import optimize
from exllamav2.conversion.compile import compile_model
from exllamav2.conversion.qparams import qparams_headoptions
import torch

parser = argparse.ArgumentParser(description = "Convert model to ExLlamaV2")
parser.add_argument("-i", "--in_dir", type = str, help = "Input directory", default = "")
parser.add_argument("-o", "--out_dir", type = str, help = "Output (working) directory")
parser.add_argument("-res", "--resume", action = "store_true", help = "Resume job from specified output directory (without specifying other options)")
parser.add_argument("-nr", "--no_resume", action = "store_true", help = "Do not resume an interrupted job (deletes all files in the output directory)")
parser.add_argument("-cf", "--compile_full", type = str, help = "Output folder for compiled model with all config/tokenizer files")
parser.add_argument("-c", "--cal_dataset", type = str, help = "Calibration dataset (.parquet file)")
parser.add_argument("-b", "--bits", type = float, default = 4.125, help = "Target bits per weight")
parser.add_argument("-ss", "--shard_size", type = float, help = "Max shard size in MB (default: 8192)", default = 8192)
parser.add_argument("-rs", "--rope_scale", type = float, help = "RoPE scaling factor")
parser.add_argument("-ra", "--rope_alpha", type = float, help = "RoPE alpha value (NTK)")
parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Target bits per weight (head layer). Use 16 to leave the head unquantized (fp16)")
parser.add_argument("-om", "--output_measurement", type = str, help = "Only perform measurement pass, then save measurement to the specified file")
parser.add_argument("-m", "--measurement", type = str, help = "Reuse previous measurement")
parser.add_argument("-r", "--dataset_rows", type = int, default = 100, help = "Number of rows to apply from dataset")
parser.add_argument("-mr", "--measurement_rows", type = int, default = 16, help = "Number of rows to apply from dataset when measuring")
parser.add_argument("-l", "--length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-ml", "--measurement_length", type = int, default = 2048, help = "Max no. tokens per sample when measuring")
parser.add_argument("-so", "--status_output", action = "store_true", help = "Include machine-parseable status updates in console output")
parser.add_argument("-hsol", "--hidden_state_offload_layers", type = int, default = 0, help = "Number of hidden/target states to keep in VRAM. Speed-up but increases VRAM usage")
parser.add_argument("-fst", "--fast_safetensors", action = "store_true", help = "Deprecated (does nothing)")
parser.add_argument("-d", "--devices", type = str, default = None,
                    help = "Comma-separated CUDA device indices to use for quantization, e.g. '0,1'. "
                           "Default: all visible devices. The first entry is the primary and owns every Hessian "
                           "inversion. Tensors are never split; independent tensors are distributed across devices. "
                           "quant-plan.md §6.4")
parser.add_argument("-dr", "--device_ratios", type = str, default = None,
                    help = "Comma-separated relative throughput weights, one per device in -d, e.g. '1,2' for two "
                           "Ampere cards where the second is twice as fast. Default: learned from measurement, seeded "
                           "by architecture prior. quant-plan.md §6.7")
parser.add_argument("-crs", "--calib_row_shards", type = str, default = "off", choices = ["on", "off"],
                    help = "Shard calibration rows across -d devices, one module replica per device. "
                           "Off by default: it needs VRAM for a replica on each device, and a device that "
                           "cannot hold one is dropped. quant-plan.md §6.5")
parser.add_argument("-ps", "--pinned_staging", type = str, default = "on", choices = ["on", "off"],
                    help = "Stage weights to the host through pinned memory on a side stream instead of a "
                           "synchronous pageable copy. Single-GPU win; produces bit-identical output. "
                           "quant-plan.md §6.8")
parser.add_argument("-x25u", "--exl25_uniform_bits", type = int, default = 0,
                    help = "EXL2.5: quantize every linear at a single bit-width (2/3/4/5/6/8) with uniform "
                           "groups, served by the branchless fast-lane GEMM (m-cap 16). 0 = legacy mixed-bit EXL2.")
parser.add_argument("-x25g", "--exl25_group_size", type = int, default = 128,
                    help = "EXL2.5: uniform group size along K (32/64/128/256; K must divide). Default 128.")
parser.add_argument("-x25h", "--exl25_hadamard", action = "store_true",
                    help = "EXL2.5: K-side Hadamard incoherence (requires --exl25_uniform_bits). Weights are "
                           "pre-rotated at conversion; the GEMM rotates activations in-kernel. Lower quant error "
                           "at the same bpw.")
parser.add_argument("-x25a", "--exl25_awq", action = "store_true",
                    help = "EXL2.5: AWQ-style per-channel protection scaling from calibration activations "
                           "(requires --exl25_uniform_bits).")

args = parser.parse_args()

torch.set_printoptions(precision = 7, sci_mode = False, linewidth = 200)

# Check some args

resuming = False
if args.out_dir:
    if not args.no_resume:
        if os.path.exists(os.path.join(args.out_dir, "job_new.json")):
            resuming = True
else:
    print(" ## Please specify output/working directory (-o, --out_dir)")
    sys.exit()

if not args.in_dir and not resuming:
    print(" ## Please specify input model directory (-i, --in_dir)")
    sys.exit()

if args.length > 2048 or args.measurement_length > 2048:
    print(" !! Warning: calibration rows > 2048 tokens may result in excessive VRAM use")

if not args.head_bits in qparams_headoptions:
    print(f" ## Error: {args.head_bits} is not a supported option for head layer bitrate")
    print(f" ##        Supported options: {sorted(qparams_headoptions.keys())} (16 = unquantized fp16 head)")
    sys.exit()

if args.output_measurement is not None and args.compile_full is not None:
    print(" ## Conflicting options: --output_measurement and --compile_full")
    sys.exit()

if args.bits < 2 or args.bits > 8:
    print(f" !! Warning: target bitrate {args.bits} will likely not be attainable")

if not os.path.exists(args.out_dir):
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f" -- Created output directory: {args.out_dir}")
    except OSError as e:
        print(f" ## Error: Failed to create output directory: {args.out_dir}")
        print(f"    {str(e)}")
        sys.exit()

# Create job

def save_job():
    global job_file, job
    with open(job_file, "w", encoding = "utf8") as f:
        f.write(json.dumps(job, indent = 4))

job_file = os.path.join(args.out_dir, "job_new.json")

if args.no_resume or not os.path.exists(job_file):

    print(f" -- Beginning new job")
    if len(os.listdir(args.out_dir)) != 0:
        print(f" !! Warning: Output directory is not empty: {args.out_dir}")

        if args.no_resume:
            print(f" !! Cleaning output directory: {args.out_dir}")
            for filename in os.listdir(args.out_dir):
                file_path = os.path.join(args.out_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

output_measurement = args.output_measurement
if output_measurement is not None:
    if os.path.isdir(output_measurement):
        output_measurement = os.path.join(output_measurement, "measurement.json")

# Device list and ratios. Parsed here so a bad value fails before any work starts.

_devices = None
if args.devices is not None:
    try:
        _devices = [int(x) for x in args.devices.replace(" ", "").split(",") if x != ""]
    except ValueError:
        print(f" ## Error: --devices expects comma-separated integers, got '{args.devices}'")
        sys.exit()
    if not _devices:
        print(f" ## Error: --devices was empty")
        sys.exit()

_device_ratios = None
if args.device_ratios is not None:
    try:
        _device_ratios = [float(x) for x in args.device_ratios.replace(" ", "").split(",") if x != ""]
    except ValueError:
        print(f" ## Error: --device_ratios expects comma-separated numbers, got '{args.device_ratios}'")
        sys.exit()
    if _devices is None:
        print(f" ## Error: --device_ratios requires --devices")
        sys.exit()
    if len(_device_ratios) != len(_devices):
        print(f" ## Error: --device_ratios has {len(_device_ratios)} entries but --devices has {len(_devices)}")
        sys.exit()
    if any(r <= 0 for r in _device_ratios):
        print(f" ## Error: --device_ratios entries must be positive")
        sys.exit()

job = {"in_dir": args.in_dir,
       "out_dir": args.out_dir,
       "cal_dataset": args.cal_dataset,
       "bits": args.bits,
       "dataset_rows": args.dataset_rows,
       "measurement_rows": args.measurement_rows,
       "length": args.length,
       "measurement_length": args.measurement_length,
       "head_bits": args.head_bits,
       "shard_size": args.shard_size if args.shard_size > 0 else 1024 ** 3,  # 1 PB = unlimited,
       "compile_full": args.compile_full,
       "rope_scale": args.rope_scale,
       "rope_alpha": args.rope_alpha,
       "output_measurement": output_measurement,
       "devices": _devices,
       "device_ratios": _device_ratios,
       "pinned_staging": args.pinned_staging == "on",
       "calib_row_shards": args.calib_row_shards == "on",
       "exl25": ({"uniform_bits": args.exl25_uniform_bits,
                 "group_size": args.exl25_group_size,
                 "hadamard": args.exl25_hadamard,
                 "awq": args.exl25_awq}
                if args.exl25_uniform_bits or args.exl25_hadamard or args.exl25_awq else None),
       "progress": "begin"}

if job["exl25"]:
    if (job["exl25"]["hadamard"] or job["exl25"]["awq"]) and not job["exl25"]["uniform_bits"]:
        print(" ## Error: --exl25_hadamard / --exl25_awq require --exl25_uniform_bits (fast lane)")
        sys.exit()
    print(f" -- EXL2.5: {job['exl25']}")

if args.measurement is not None:
    with open(args.measurement, "r", encoding = "utf8") as f:
        imp_measurement = json.load(f)
        job["measurement"] = imp_measurement["measurement"]
        job["last_module_idx"] = imp_measurement["last_module_idx"]
        job["reuse_measurement"] = args.measurement

# Resume existing job

if args.no_resume or not os.path.exists(job_file):
    pass

else:
    print(f" -- Resuming job")
    if args.in_dir:
        print(f" !! Note: Overriding options with settings from existing job")

    with open(job_file, "r", encoding = "utf8") as f:
        resume_job = json.load(f)

    # Override keys in existing job
    del resume_job["out_dir"]

    job.update(resume_job)
    if "invalid" in job:
        print(" ** Error: Corrupted job")
        sys.exit()

    if job["progress"] == "finished":
        print(f" !! Job is already finished. Clear the working directory, or run this script with -nr/--no_resume to clear it automatically.")
        sys.exit()

# Feedback

print(f" -- Input: {job['in_dir']}")
print(f" -- Output: {job['out_dir']}")
if job.get("cal_dataset"):
    print(f" -- Calibration dataset: {job['cal_dataset']}, {job['dataset_rows']} / {job['measurement_rows']} rows, {job['length']} tokens per sample")
else:
    print(f" -- Using default calibration dataset")
if job["output_measurement"] is None:
    print(f" -- Target bits per weight: {job['bits']} (decoder), {job['head_bits']} (head)")
    print(f" -- Max shard size: {job['shard_size']} MB")
else:
    print(f" -- Measurement will be saved to {job['output_measurement']}")
    print(f" !! Conversion script will end after measurement pass")

if job['rope_scale']: print(f" -- RoPE scale: {job['rope_scale']:.2f}")
if job['rope_alpha']: print(f" -- RoPE alpha: {job['rope_alpha']:.2f}")

# Seed the learned device split from an explicit ratio if one was given, so the very first tensor is already
# balanced. Without -dr the split starts from the architecture prior and refines from measurement.

if job.get("device_ratios") and job.get("devices"):
    from exllamav2.conversion.devices import auto_split
    for kind in ["attn", "mlp", "moe"]:
        for dev, r in zip(job["devices"], job["device_ratios"]):
            auto_split.speeds.setdefault(kind, {})[dev] = r
    print(f" -- Device ratios (fixed): {job['device_ratios']}")
elif job.get("devices") and len(job["devices"]) > 1:
    from exllamav2.conversion.devices import arch_prior_speed
    priors = [arch_prior_speed(d) for d in job["devices"]]
    print(f" -- Device ratios (architecture prior): {priors}")

# Make sure subfolders exist

if job.get("compile_full"):
    print(f" -- Full model will be compiled to: {job['compile_full']}")
    if os.path.exists(job["compile_full"]):
        if not os.path.isdir(job["compile_full"]):
            print(f" ## Error: Output path {job['compile_full']} exists but is not a directory")
            sys.exit()
        if len(os.listdir(job["compile_full"])) > 0:
            print(f" !! Warning: Output path {job['compile_full']} exists but is not empty")

out_tensor_dir = os.path.join(job["out_dir"], "out_tensor")
if not os.path.exists(out_tensor_dir):
    os.makedirs(out_tensor_dir)

# Create config

config = ExLlamaV2Config()
config.model_dir = job['in_dir']
config.prepare()
config.arch_compat_overrides()

# Tokenizer

tokenizer = ExLlamaV2Tokenizer(config)

# Set scaling for input model

if job["rope_scale"] is not None: config.scale_pos_emb = job["rope_scale"]
if job["rope_alpha"] is not None: config.scale_alpha_value = job["rope_alpha"]

# Create model without loading weights

model = ExLlamaV2(config)
model.load(lazy = True)

# Limit context length if necessary

if model.config.arch.lm.rope_style == RopeStyle.NONE:
    max_ctx = model.config.max_seq_len
    if job["length"] > max_ctx:
        print (f" !! Warning: Reducing calibration length to model max context: {max_ctx}")
        job["length"] = max_ctx
    if job["measurement_length"] > max_ctx:
        print (f" !! Warning: Reducing measurement calibration length to model max context: {max_ctx}")
        job["measurement_length"] = max_ctx

# Overridable settings

job["status_output"] = args.status_output

# Do the things

save_job()

while True:

    progress = job["progress"]

    if progress == "begin":

        if "reuse_measurement" in job:

            print(f" -- Reusing measurement: {job['reuse_measurement']}")
            job["progress"] = "optimize"
            save_job()

        else:

            print(f" -- Tokenizing samples (measurement)...")
            noise_rows = config.arch.standard_calib_noise
            tokenize(job, save_job, tokenizer, measure = True, noise_rows = noise_rows)
            job["progress"] = "initial_embeddings"
            save_job()

    if progress == "initial_embeddings":

        print(f" -- Token embeddings (measurement)...")
        embeddings(job, save_job, model)
        job["progress"] = "measure_quant"
        save_job()

    if progress == "measure_quant":
        print(f" -- Measuring quantization impact...")

        model.unload()
        config.max_output_len = 16
        model = ExLlamaV2(config)
        model.load(lazy = True)

        status = measure_quant(job, save_job, model, args.hidden_state_offload_layers)  # capturing the graceful exits
        if status == "interrupted":
            print("Process interrupted. Exiting gracefully.")
            save_job()
            sys.exit(1)  
        if job["output_measurement"] is None:
            job["progress"] = "optimize"
        else:
            job["progress"] = "finished"
        save_job()

        model.unload()
        config.max_output_len = None
        model = ExLlamaV2(config)
        model.load(lazy = True)

    if progress == "optimize":

        print(f" -- Optimizing...")
        optimize(job, save_job, model)
        job["progress"] = "tokens_cal"
        save_job()

    if progress == "tokens_cal":

        print(f" -- Tokenizing samples...")
        noise_rows = config.arch.standard_calib_noise
        tokenize(job, save_job, tokenizer, noise_rows = noise_rows)
        job["progress"] = "embeddings"
        save_job()

    if progress == "embeddings":
        print(f" -- Token embeddings again...")
        embeddings(job, save_job, model)
        job["progress"] = "quant"
        save_job()

    if progress == "quant":

        print(f" -- Quantizing...")
        quant(job, save_job, model)
        job["progress"] = "compile"
        save_job()

    if progress == "compile":

        print(f" -- Compiling output file...")
        compile_model(job, save_job, model)
        job["progress"] = "finished"
        save_job()

    if progress == "finished": break

print(f" -- Finished")





