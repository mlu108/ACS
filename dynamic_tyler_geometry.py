

#import circuitsvis as cv

import torch
from tqdm import tqdm

from transformer_lens import HookedEncoder, ActivationCache,HookedTransformer
from transformer_lens import patching
import transformer_lens.utils as utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float
from typing import Callable
from functools import partial
import itertools
import os
import numpy as np
import json
import pandas as pd

import torch.nn.functional as F
import csv

import plotly.express as px

from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from jaxtyping import Float

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer
from helpers import load_hooked_transformer_model
torch.set_grad_enabled(False)
device = utils.get_device()


#INPUT:
import argparse

available_models = ['llama-1b','llama-3b','llama-7b','llama-8b','Qwen2_1.5b','Qwen2-0.5B','Qwen2.5-14B']

# parse_known_args (not parse_args) so this still runs unmodified under a
# Jupyter kernel, which injects its own argv (e.g. "-f kernel-xxx.json") that
# would otherwise make argparse error out and kill the kernel.
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--model_name", type=str, default=available_models[-1])
_args, _unknown_argv = _arg_parser.parse_known_args()
MODEL = _args.model_name

print(MODEL)
model,tokenizer = load_hooked_transformer_model(MODEL)
print(model.cfg.n_layers)



import gc
import torch
import math
import os
import re
import torch
import codecs
from datasets import load_dataset
LANGUAGE_CODE_MAP = {
    "english": "en",
    "chinese": "zh",
    "french": "fr",
    "japanese": "ja",
    "korean": "ko",
    "spanish": "es",
    "catalan": "ca",
    "hungarian": "hu",
    "dutch": "nl",
    "russian": "ru",
    "ukrainian": "uk",
    "vietnamese": "vi"
}

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)


from datasets import load_dataset

def get_oscar_sequences_for_lang(lang, model, device, 
                                 n_sequences=600, 
                                 tokens_per_seq=200, 
                                 use_auth_token=True):
    """
    Stream text from OSCAR 'deduplicated_<lang>', gather exactly `n_sequences` 
    each of length ~ tokens_per_seq. Returns a list of strings (one per sequence).
    
    Now prints 3 preview examples before tokenization.
    """
    # --- Language code mapping ---
    lang_code = LANGUAGE_CODE_MAP[lang]  # e.g. "english" -> "en"
    dataset_subset = f"deduplicated_{lang_code}"

    # --- Load dataset stream ---
    ds = load_dataset(
        "oscar-corpus/OSCAR-2109",
        dataset_subset,
        split="train",
        streaming=True,
        token=os.environ["HF_TOKEN"],
        cache_dir="hf_datasets_cache",
        trust_remote_code=True,
    )

    # --- Preview 3 samples before processing ---
    print(f"\n=== Previewing 3 OSCAR samples for {lang} ({dataset_subset}) ===")
    preview_count = 0
    for example in ds:
        text = example.get("text", "").strip()
        if text:
            print(f"[{preview_count+1}] {text[:400]}")
            preview_count += 1
        if preview_count >= 3:
            break
    print("=== End of preview ===\n")

    # --- Reset stream after preview (create new iterator) ---
    ds = load_dataset(
        "oscar-corpus/OSCAR-2109",
        dataset_subset,
        split="train",
        streaming=True,
        token=os.environ["HF_TOKEN"],
        cache_dir="hf_datasets_cache",
        trust_remote_code=True,
    )

    # --- Token accumulation ---
    sequences = []
    token_buffer = []

    for example in ds:
        text = example.get("text", "").strip()
        if not text:
            continue

        # Minimal cleaning
        text = text.replace("\\n", " ").replace("\\t", " ")

        encoded_tokens = model.to_tokens(text).to(device)  # shape (1, seq_len)
        encoded_tokens = encoded_tokens.squeeze(0)         # shape (seq_len,)

        # Extend local buffer
        token_buffer.extend(encoded_tokens.tolist())

        while len(token_buffer) >= tokens_per_seq:
            seq_ids = token_buffer[:tokens_per_seq]
            token_buffer = token_buffer[tokens_per_seq:]
            seq_str = model.to_string(
                torch.tensor(seq_ids, device=device).unsqueeze(0)
            )
            sequences.append(seq_str)
            if len(sequences) >= n_sequences:
                return sequences

    return sequences  # if dataset exhausted early


import torch
import os
from tqdm import tqdm


def all_one_lan_data_resid_pre(LANGUAGE, LAYER, model, device,n_sequences=1200,tokens_per_seq=200):
    """
    1) Fetch ~N sequences from OSCAR in the specified language (defaults: 600 sequences of 200 tokens).
    2) For each sequence, run the model once and collect `resid_pre` for *every token position* at layer LAYER.
    3) Return a stacked tensor of shape (N * seq_len, d_model).
       E.g., if N=600 and seq_len=200, final shape is (120000, d_model).
    """
    all_LAYER_embeddings = []
    
    # 1) Fetch sequences (here, 600 sequences, each 200 tokens)
    sequences = get_oscar_sequences_for_lang(
        LANGUAGE, model, device,
        n_sequences=n_sequences,
        tokens_per_seq=tokens_per_seq
    )
    
    if len(sequences) == 0:
        print(f"No sequences found for {LANGUAGE}.")
        return None
    
    # 2) Forward pass for each sequence and store resid_pre for *all token positions*
    for seq_str in tqdm(sequences, desc=f"Collecting *all positions* embeddings for {LANGUAGE} / layer {LAYER}"):
        # tokens: shape (1, seq_len)
        tokens = model.to_tokens(seq_str).to(device)
        
        # run_with_cache => get the hidden states
        _, cache = model.run_with_cache(tokens, return_type="logits")
        
        # The `resid_pre` at layer LAYER has shape (batch_size=1, seq_len, d_model)
        layer_name = utils.get_act_name("resid_pre", LAYER)  # e.g. "blocks.0.hook_resid_pre"
        resid_pre_seq = cache[layer_name][0, :, :]  # shape: (seq_len, d_model)
        
        # Collect for this sequence
        all_LAYER_embeddings.append(resid_pre_seq)
    
    # 3) Concatenate along the sequence dimension to get (N * seq_len, d_model)
    if len(all_LAYER_embeddings) > 0:
        all_LAYER_embeddings_tensor = torch.cat(all_LAYER_embeddings, dim=0)         
        return all_LAYER_embeddings_tensor
    else:
        print(f"No embeddings collected for {LANGUAGE} at layer {LAYER}.")
        return None
import torch
import torch
import math
import os
import re
import codecs

from datasets import load_dataset




languages = ['french','japanese','korean','spanish','english','chinese', 'hungarian', 'dutch', 'russian', 'ukrainian','vietnamese']
#valid_langs = ["ca",  "hu", "nl", "ru", "uk", "vi"]#"en","es", "fr","ja", "ko",  "zh"

layers = range(model.cfg.n_layers)  # e.g., layers 0 to 31 for a 32-layer model

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for LAYER in layers:
    print(f"Processing LAYER: {LAYER}")
    for LANGUAGE in languages:
        # Get the embeddings for each language from the OSCAR corpus
        try:
            all_LAYER_embeddings_tensor = all_one_lan_data_resid_pre(LANGUAGE, LAYER, model, device,
            n_sequences=200,tokens_per_seq=20)
        
        
            # Directory to save the embeddings
            save_dir = f"oscar_geometry_{MODEL}/all_one_lan_data_resid_pre_oscar2109/{LANGUAGE}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the tensor as a .pt file
            if all_LAYER_embeddings_tensor is not None:
                save_path = os.path.join(save_dir, f"LAYER_{LAYER}_{LANGUAGE}.pt")
                torch.save(all_LAYER_embeddings_tensor, save_path)
                print(f"Saved embeddings to: {save_path} | Shape: {all_LAYER_embeddings_tensor.shape}")
                del all_LAYER_embeddings_tensor
                gc.collect()
                torch.cuda.empty_cache()

            else:
                print(f"No embeddings collected for LANGUAGE: {LANGUAGE} at LAYER: {LAYER}")
        except Exception as e:
            print(f"Error processing LANGUAGE: {LANGUAGE} at LAYER: {LAYER} | Error: {e}")
            continue
