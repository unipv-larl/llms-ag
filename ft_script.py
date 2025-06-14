import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import datasets
import deepspeed
import bitsandbytes
import trl
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState
from huggingface_hub import login
from datasets import load_dataset
from trl import SFTTrainer
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

my_key = os.environ.get('MY_KEY')
login(token=my_key)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-Nemo-Instruct-2407",
    device_map='auto',
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407", add_prefix_space=True, padding_side='right')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.use_cache = False

dataset = load_dataset('json', data_files={
    'train': 'train.jsonl',
    'validation': 'validate.jsonl'
})

def tokenize_function(examples):
    tokenizer.truncation_side = "right"
    inputs = [f"Input: {i}\nOutput: {o}" for i, o in zip(examples["input"], examples["synonyms"])]

    tokenized_inputs = tokenizer(
        inputs,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=120
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input", "synonyms"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

metrics_history = {
    'epoch': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'eval_loss': [],
    'perplexity': []
}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = torch.tensor(predictions[0])
    labels = torch.tensor(labels)
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), predictions.flatten(), average='weighted')
    eval_loss = trainer.state.log_history[-1].get('eval_loss', None)
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    current_epoch = trainer.state.epoch if trainer.state.epoch is not None else len(metrics_history['epoch']) + 1

    metrics_history['epoch'].append(current_epoch)
    metrics_history['accuracy'].append(accuracy)
    metrics_history['precision'].append(precision)
    metrics_history['recall'].append(recall)
    metrics_history['f1'].append(f1)
    metrics_history['eval_loss'].append(eval_loss)
    metrics_history['perplexity'].append(perplexity)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eval_loss': eval_loss,
        'perplexity': perplexity
    }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

args = TrainingArguments(
    output_dir='.venv/results',
    learning_rate= 1e-4,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    metric_for_best_model='eval_loss',
    fp16=True,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    logging_steps=1,
    torch_compile=True,
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=args,
    processing_class=tokenizer,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

model.train()

wandb_key = os.environ.get('WANDB_KEY')

wandb.login(key=wandb_key)

torch.set_grad_enabled(True)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

trainer.train()

model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

results = trainer.evaluate()
def save_metric_plots(metrics_dict, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    for metric, values in metrics_dict.items():
        if not values or values[0] is None:
            continue
        plt.figure()
        plt.plot(range(1, len(values) + 1), values, marker='o')
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()

save_metric_plots(metrics_history)

print(results)
print('Fine-tuning completed!')
