import argparse
import json
import os
import re

import torch

from config import (
    DEFAULT_AC_PRECISION,
    DEFAULT_DISCOP_BASELINE,
    DEFAULT_IMEC_BLOCK_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_METHOD,
    DEFAULT_METEOR_REORDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PRECISION,
    DEFAULT_SEED,
    DEFAULT_SINGLE_INPUT_TEXT,
    DEFAULT_SINGLE_MESSAGE_BITS,
    DEFAULT_SPARSAMP_BLOCK_SIZE,
    DEFAULT_TEMP,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SUPPORTED_METHODS,
    SUPPORTED_MODEL_NAMES,
    get_single_output_path,
)
from utils import (
    decode_with_method,
    encode_context,
    encode_with_method,
    initialize_model_and_precision,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-sample steganography + extraction")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=SUPPORTED_METHODS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--input_text", type=str, default=DEFAULT_SINGLE_INPUT_TEXT, help="Prompt/context sentence")
    parser.add_argument("--message_bits", type=str, default=DEFAULT_SINGLE_MESSAGE_BITS, help="Bit string to embed, e.g. 010101")

    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model_precision", type=str, default=DEFAULT_MODEL_PRECISION)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--ac_precision", type=int, default=DEFAULT_AC_PRECISION)
    parser.add_argument("--discop_baseline", action="store_true", default=DEFAULT_DISCOP_BASELINE)
    parser.add_argument("--imec_block_size", type=int, default=DEFAULT_IMEC_BLOCK_SIZE)
    parser.add_argument("--meteor_reorder", action="store_true", default=DEFAULT_METEOR_REORDER)
    parser.add_argument("--sparsamp_block_size", type=int, default=DEFAULT_SPARSAMP_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def validate_message_bits(message_bits: str):
    if not message_bits:
        raise ValueError("--message_bits cannot be empty.")
    if not re.fullmatch(r"[01]+", message_bits):
        raise ValueError("--message_bits must contain only '0' and '1'.")


def load_last_jsonl_record(path: str):
    last_record = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                last_record = obj
    if last_record is None:
        raise ValueError(f"No valid JSONL records found in {path}")
    return last_record


def main():
    args = parse_args()
    validate_message_bits(args.message_bits)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    enc, model = initialize_model_and_precision(args, seed=args.seed)
    model.to(device)
    model.eval()

    context_tokens = encode_context(args.input_text, enc)
    if not context_tokens:
        raise ValueError("Input prompt tokenization is empty.")

    generated_ids, embedded_bits = encode_with_method(
        args.method,
        model,
        context_tokens,
        args.message_bits,
        args,
        device,
    )

    stego_text = enc.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    output_path = get_single_output_path(args.method)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jsonl_record = {
        "model_name": args.model_name,
        "input_text": args.input_text,
        "message_bits": args.message_bits,
        "text": stego_text,
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")

    stego_text_from_file = str(load_last_jsonl_record(output_path).get("text", ""))
    if not stego_text_from_file:
        raise ValueError("The last JSONL record does not contain a valid 'text' field.")

    generated_ids_from_file = enc.encode(stego_text_from_file, add_special_tokens=False)
    extracted_bits = decode_with_method(
        args.method,
        model,
        context_tokens,
        generated_ids_from_file,
        args,
        device,
    )

    is_match = extracted_bits.startswith(embedded_bits) if embedded_bits else True

    print(f"Stego JSONL appended to: {output_path}")
    print(f"Embedded bits length: {len(embedded_bits)}")
    print(f"Extracted bits length: {len(extracted_bits)}")
    print(f"Extraction prefix match: {is_match}")
    print(f"Embedded bits: {embedded_bits}")
    print(f"Extracted bits: {extracted_bits}")


if __name__ == "__main__":
    main()
