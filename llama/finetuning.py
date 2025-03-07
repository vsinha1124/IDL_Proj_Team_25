from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import time

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define test prompts to use before and after fine-tuning
test_prompts = [
    "Give three tips for staying healthy.",
    "How can I improve my productivity?",
    "Explain quantum computing in simple terms."
]

# Function to test model with prompts
def test_model(model, tokenizer, prompts, prefix=""):
    print(f"\n{prefix} TESTING MODEL RESPONSES:")
    print("=" * 50)
    
    results = {}
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        print("-" * 50)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with the model
        with torch.no_grad():
            gen_start = time.time()
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_length=256,
                temperature=0.7,
                top_p=0.9,
                top_k=40
            )
            gen_time = time.time() - gen_start
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print(f"Generation time: {gen_time:.2f} seconds")
        
        results[prompt] = response
    
    print("=" * 50)
    return results

# Model name
model_name = "meta-llama/Llama-3.2-3B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with proper config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded successfully!")

# Test model BEFORE fine-tuning
before_results = test_model(model, tokenizer, test_prompts, prefix="BEFORE FINE-TUNING")

# Now set use_cache=False for training
model.config.use_cache = False

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca")
print(f"Full dataset loaded with {len(dataset['train'])} examples")

# Create a subset - just the first 100 samples
subset_size = 100
dataset["train"] = dataset["train"].select(range(subset_size))
print(f"Created subset with {len(dataset['train'])} examples")
print(f"Sample: {dataset['train'][0]}")

# Apply LoRA fine-tuning with adjusted config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("Model prepared for training.")

# Print trainable parameters
trainable_params = 0
all_params = 0
for _, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)")

# Tokenize dataset with proper formatting
def tokenize_function(examples):
    # Format prompts for instruction tuning
    prompts = [
        f"### Instruction:\n{instruction}\n### Input:\n{inp}\n### Response:\n{output}"
        for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,  # We'll pad in the data collator
        max_length=512
    )
    
    # Important: Make labels a copy of input_ids, not integers
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Process the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "input", "output", "text"]
)

print("Dataset tokenized successfully.")
print(f"Sample tokenized data: {tokenized_dataset['train'][0]}")

# Verify label structure to make sure they're not integers
sample = tokenized_dataset["train"][0]
print(f"Labels type: {type(sample['labels'])}")
print(f"Labels length: {len(sample['labels'])}")

# Use DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training Arguments with adjusted values
training_args = TrainingArguments(
    output_dir="./fine-tuned-llama-subset",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=20,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    report_to="none",
    warmup_steps=10,
    weight_decay=0.01
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=training_args,
    data_collator=data_collator,
)

# Run training
try:
    print(f"Starting training on {len(tokenized_dataset['train'])} examples...")
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    print("\nDebugging information:")
    batch = [tokenized_dataset["train"][i] for i in range(min(4, len(tokenized_dataset["train"])))]
    print(f"Batch keys: {batch[0].keys()}")

# If training completes, save the model
try:
    print("Saving model...")
    peft_model_id = "./fine-tuned-llama-subset"
    model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    print("Model saved successfully!")
    
    # Test the fine-tuned model
    print("\nTesting the fine-tuned model:")
    
    # For proper comparison, we need to load a fresh base model and apply the saved LoRA adapters
    from peft import PeftModel, PeftConfig
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load the LoRA adapters
    fine_tuned_model = PeftModel.from_pretrained(base_model, peft_model_id)
    
    # Test the fine-tuned model with the same prompts
    after_results = test_model(fine_tuned_model, tokenizer, test_prompts, prefix="AFTER FINE-TUNING")
    
    # Compare the results
    print("\nCOMPARISON OF RESPONSES:")
    print("=" * 80)
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        print("-" * 80)
        print(f"BEFORE: {before_results[prompt]}")
        print("-" * 40)
        print(f"AFTER: {after_results[prompt]}")
        print("-" * 80)
    
except Exception as save_err:
    print(f"Error saving or testing model: {save_err}")