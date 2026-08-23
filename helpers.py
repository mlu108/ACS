
import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import login
from transformer_lens.model_bridge import TransformerBridge


def _maybe_login_hf():
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)

def load_hooked_transformer_model(model_name):
    name = model_name.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    load_in_dtype = False

    _maybe_login_hf()

    if "llama" in name:
        if "1b" in name:
            model_id = "meta-llama/Llama-3.2-1B-Instruct" if "instruct" in name else "meta-llama/Llama-3.2-1B"
        elif "3b" in name:
            model_id = "meta-llama/Llama-3.2-3B-Instruct" if "instruct" in name else "meta-llama/Llama-3.2-3B"
        elif "7b" in name:
            model_id = "meta-llama/Llama-2-7b-hf"
        elif "8b" in name:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError(f"Unknown llama model: {model_name}")

    elif "qwen" in name:
        if "1.5" in name:
            model_id = "Qwen/Qwen2-1.5B-Instruct"
        elif "0.5" in name:
            model_id = "Qwen/Qwen2-0.5B"
        elif "14" in name:
            model_id = "Qwen/Qwen2.5-14B"
            # ~29.5GB in bf16 — fits on a single 48GB card (e.g. l40s) but
            # NOT in the default float32 load (~59GB), so force bf16 rather
            # than sharding across GPUs (cross-GPU hook dispatch was both
            # much slower and producing garbled generations).
            load_in_dtype = True
        elif "3" in name:
            model_id = "Qwen/Qwen2.5-3B"
        else:
            raise ValueError(f"Unknown Qwen model: {model_name}")

    elif "bloom" in name:
        model_id = "bigscience/bloom-560m"

    elif "olmo" in name:
        if "1b" in name:
            model_id = "allenai/OLMo-2-0425-1B"
        elif "7b" in name:
            model_id = "allenai/OLMo-2-1124-7B"
            # ~14GB in bf16 vs ~28GB in default float32 — the latter OOMs
            # on a 22GB GPU, so force bf16 like the Qwen-14B branch above.
            load_in_dtype = True
        else:
            raise ValueError(f"Unknown OLMo model: {model_name}")

    else:
        raise ValueError(f"MODEL NAME not configured: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if load_in_dtype and device == "cuda":
        model = TransformerBridge.boot_transformers(
            model_id,
            device=device,
            dtype=dtype,
        )
    else:
        model = TransformerBridge.boot_transformers(
            model_id,
            device=device,
            #torch_dtype=dtype,
            #trust_remote_code=True,
        )

    return model, tokenizer


def load_hooked_transformer_thinking_model(model_name):
    """
    Loads hybrid instruct models that support explicit chain-of-thought
    ("thinking") generation via the tokenizer's chat template
    (enable_thinking=True/False), e.g. Qwen3's dense checkpoints.

    Kept separate from load_hooked_transformer_model since these models
    are always used with the chat template (not raw completion prompts),
    unlike the base/instruct models loaded above.
    """
    name = model_name.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _maybe_login_hf()

    if "qwen3" in name:
        if "4b" in name:
            model_id = "Qwen/Qwen3-4B"
        else:
            raise ValueError(f"Unknown Qwen3 thinking model: {model_name}")
    else:
        raise ValueError(f"MODEL NAME not configured for thinking: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = TransformerBridge.boot_transformers(
        model_id,
        device=device,
        dtype=dtype,
    )

    return model, tokenizer