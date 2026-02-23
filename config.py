"""Project-wide constants and path helpers for steganography scripts."""

import os


# Steganography methods that support both embedding and extraction.
SUPPORTED_METHODS = ("ac", "meteor", "adg", "discop", "imec", "sparsamp")

# Canonical local model paths.
MODEL_PATH_MAP = {
    "gpt2": "YOUR_PATH/openai-community/gpt2/",
    "qwen2.5": "YOUR_PATH/Qwen/Qwen2.5-3B-Instruct/",
    "llama3.2": "YOUR_PATH/meta-llama/Llama-3.2-3B-Instruct/",
}
SUPPORTED_MODEL_NAMES = tuple(MODEL_PATH_MAP.keys())

# Shared argument defaults.
DEFAULT_METHOD = "ac"
DEFAULT_MODEL_NAME = "qwen2.5"
DEFAULT_MODEL_PRECISION = "float16"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_AC_PRECISION = None
DEFAULT_DISCOP_BASELINE = False
DEFAULT_IMEC_BLOCK_SIZE = 4
DEFAULT_METEOR_REORDER = False
DEFAULT_SPARSAMP_BLOCK_SIZE = 32
DEFAULT_SEED = 42

DEFAULT_INPUT_CSV = "context_movie.csv"
DEFAULT_MAX_CONTEXTS = 100
DEFAULT_MESSAGE_BITS_LENGTH = 0
DEFAULT_SINGLE_INPUT_TEXT = "Tell a short story about a rabbit."
DEFAULT_SINGLE_MESSAGE_BITS = "010101110011"


def resolve_model_path(model_name: str, model_path: str = "") -> str:
    """Resolve model path."""
    resolved = MODEL_PATH_MAP.get(model_name, model_path)
    if not resolved:
        raise ValueError("Please provide --model_path or use a supported --model_name.")
    return resolved


def infer_dataset_name(input_csv: str) -> str:
    """Infer dataset name."""
    file_stem = os.path.splitext(os.path.basename(input_csv))[0]
    parts = file_stem.split("_")
    if len(parts) > 1:
        return parts[1]
    return file_stem


def get_parallel_paths(input_csv: str, method: str):
    """Get parallel paths."""
    dataset = infer_dataset_name(input_csv)
    output_dir = os.path.join("results_parallel", dataset)
    tmp_dir = os.path.join(output_dir, f".tmp_{method}")
    output_jsonl_path = os.path.join(output_dir, f"{method}.jsonl")
    return dataset, output_dir, tmp_dir, output_jsonl_path


def get_single_output_path(method: str) -> str:
    """Get single output path."""
    return os.path.join("results_single", f"{method}.jsonl")
