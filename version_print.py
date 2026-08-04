import importlib

modules = [
    "python",
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "unsloth",
    "xformers",
    "triton",
    "flash_attn",
    "sentencepiece",
    "tokenizers",
    "huggingface_hub",
    "numpy",
]

print("=" * 50)
print("Environment Versions")
print("=" * 50)

for module in modules:
    try:
        if module == "python":
            import sys
            version = sys.version.split()[0]
        else:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "Version attribute not found")
        print(f"{module:<20} {version}")
    except ImportError:
        print(f"{module:<20} Not installed")