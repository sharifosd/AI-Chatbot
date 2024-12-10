from unsloth import FastLanguageModel
import torch
import json

# Load your FAQ dataset
with open('University_Faq.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# Convert FAQ data to conversation format
conversations = []
for qa in faq_data:
    conversation = [
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]}
    ]
    conversations.append({"conversations": conversation})

print(f"Total number of QA pairs: {len(conversations)}")

# Initialize model and tokenizer
max_seq_length = 2048
dtype = None 
load_in_4bit = True

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("Model and tokenizer loaded successfully!")

# Setup PEFT configuration
print("Setting up PEFT configuration...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Setup tokenizer with chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# Create dataset
from datasets import Dataset
dataset = Dataset.from_list(conversations)
print(f"Dataset created with {len(dataset)} examples")

# Training arguments
from transformers import TrainingArguments

# Calculate training parameters
num_epochs = 32
total_steps = (len(dataset) * num_epochs) // (4 * 4)  # batch_size * gradient_accumulation_steps

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir = "university_faq_model",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    warmup_steps = total_steps // 10,  # 10% of total steps
    max_steps = total_steps,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    save_strategy = "steps",
    save_steps = total_steps // 5,  # Save 5 checkpoints during training
    report_to = "tensorboard",  # Enable tensorboard logging
)

# Initialize trainer
from trl import SFTTrainer
from transformers import DataCollatorForSeq2Seq

print("Initializing trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "conversations",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = training_args,
)

# Apply response-only training
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Function to test the model
def test_model(question, model, tokenizer):
    inputs = tokenizer(
        [{"role": "user", "content": question}],
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Start training
print("Starting training...")
trainer_stats = trainer.train()
print("Training completed!")

# Save the model
print("Saving model...")
model.save_pretrained("university_faq_model")
tokenizer.save_pretrained("university_faq_model")
print("Model saved successfully!")

## Update the test_model function
def test_model(question, model, tokenizer):
    # Format the conversation using the chat template
    conversation = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test section remains the same
print("\nTesting the trained model:")
test_questions = [
    "What departments are available?",
    "What is the semester fee for Computer Science?",
    "কোন কোন ডিপার্টমেন্ট আছে?",  # Bengali test
]

for question in test_questions:
    print("\nQuestion:", question)
    response = test_model(question, model, tokenizer)
    print("Response:", response)