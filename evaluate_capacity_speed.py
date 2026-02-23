import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from pss.arithmetic import encode_arithmetic
from pss.meteor import encode_meteor
from pss.sparsamp import encode_spar
from config import SUPPORTED_MODEL_NAMES, resolve_model_path
from utils import encode_context, get_model


def infer_dataset_name(input_csv: str) -> str:
    file_stem = os.path.splitext(os.path.basename(input_csv))[0]
    parts = file_stem.split("_")
    if len(parts) > 1:
        return parts[1]
    return file_stem


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate embedding capacity and speed for steganography methods")
    parser.add_argument("--model_name", type=str, default="qwen2.5", choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--model_precision", type=str, default="float16")
    parser.add_argument("--input_csv", type=str, default="context_movie.csv")
    parser.add_argument("--dataset", type=str, default="", help="Default inferred from --input_csv")
    parser.add_argument("--max_contexts", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--ac_precision", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--meteor_reorder", action="store_true")
    parser.add_argument("--sparsamp_block_size", type=int, default=32)
    return parser.parse_args()


def load_contexts(path: str, max_contexts: int):
    contexts = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                contexts.append(row[1])
    if max_contexts > 0:
        contexts = contexts[:max_contexts]
    return contexts


def main():
    args = parse_args()

    model_path = resolve_model_path(args.model_name)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading model: {args.model_name} ({args.model_precision}) on {device}...")
    enc, model, precision_int = get_model(
        model_name=model_path,
        model_precision=args.model_precision,
        seed=args.seed,
    )
    if args.ac_precision is None:
        args.ac_precision = precision_int
    model.to(device)
    model.eval()
    print("Model loaded.")

    contexts_to_process = load_contexts(args.input_csv, args.max_contexts)
    print(f"Loaded {len(contexts_to_process)} contexts from {args.input_csv}")

    method_dispatcher = {
        "ac": encode_arithmetic,
        "meteor": encode_meteor,
        "sparsamp": encode_spar,
    }

    final_results = []

    for method_name, encode_function in method_dispatcher.items():
        total_capacity = 0.0
        total_speed = 0.0
        processed_count = 0

        print(f"\n--- Processing method: {method_name} ---")

        for context_text in tqdm(contexts_to_process, desc=f"Method {method_name}"):
            try:
                context_tokens = encode_context(context_text, enc)
                if not context_tokens:
                    continue

                message_bits = "".join(random.choice("01") for _ in range(args.max_tokens * 16))

                if method_name == "ac":
                    output = encode_function(
                        model,
                        message_bits,
                        context_tokens,
                        args.max_tokens,
                        device=device,
                        precision=args.ac_precision,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        temp=args.temp,
                    )
                elif method_name == "meteor":
                    output = encode_function(
                        model,
                        context_tokens,
                        message_bits,
                        args.max_tokens,
                        device=device,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        precision=args.ac_precision,
                        reorder=args.meteor_reorder,
                        temp=args.temp,
                    )
                elif method_name == "sparsamp":
                    output = encode_function(
                        model,
                        context_tokens,
                        message_bits,
                        args.max_tokens,
                        device=device,
                        block_size=args.sparsamp_block_size,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        random_seed=args.seed,
                        temp=args.temp,
                    )
                else:
                    continue

                generated_ids, encoded_message, _, _, elapsed_time = output

                bits_embedded = len("".join(encoded_message)) if isinstance(encoded_message, list) else len(encoded_message)
                num_generated_tokens = len(generated_ids)

                if num_generated_tokens > 0 and elapsed_time > 0:
                    total_capacity += bits_embedded / num_generated_tokens
                    total_speed += bits_embedded / elapsed_time
                    processed_count += 1

            except Exception as exc:
                tqdm.write(f"Skipping one sample for {method_name}: {exc}")

        if processed_count > 0:
            avg_capacity = total_capacity / processed_count
            avg_speed = total_speed / processed_count
        else:
            avg_capacity = 0.0
            avg_speed = 0.0

        final_results.append(
            {
                "method": method_name,
                "avg_capacity": avg_capacity,
                "avg_speed": avg_speed,
                "processed_count": processed_count,
            }
        )
        print(
            f"Method {method_name}: Avg. Capacity = {avg_capacity:.4f} bits/word, "
            f"Avg. Speed = {avg_speed:.2f} bits/s"
        )

    dataset = args.dataset or infer_dataset_name(args.input_csv)
    output_dir = os.path.join("results_parallel", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"capacity_speed_{args.model_name}.jsonl")

    print("\nFinal Averaged Results")
    with open(output_file, "w", encoding="utf-8") as f:
        for res in final_results:
            record = {
                "model_name": args.model_name,
                "dataset": dataset,
                "input_csv": args.input_csv,
                "max_contexts": args.max_contexts,
                "max_tokens": args.max_tokens,
                "method": res["method"],
                "avg_capacity": float(res["avg_capacity"]),
                "avg_speed": float(res["avg_speed"]),
                "processed_count": int(res["processed_count"]),
            }
            print(
                f"{record['method']}: avg_capacity={record['avg_capacity']:.4f}, "
                f"avg_speed={record['avg_speed']:.2f}, processed={record['processed_count']}"
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
