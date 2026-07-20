from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-12b-it",
    max_seq_length=4096,
    load_in_4bit=True,
)

print("Gemma 4 loaded successfully!")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

text = "Hello, how are you?"

inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=32,
)

print(tokenizer.decode(outputs[0]))