# from datasets import load_dataset
# from trl import DPOConfig, DPOTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import wandb

# model_name = "Qwen2.5-Coder-0.5B-Instruct"
# train_data_file = "/Users/tbassman/Desktop/GitHub/External/cs229/project/moatless-tree-search/moatless/20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8_trajectories_preference.csv"
# exp_dir = f"./{model_name}_DPO_0"

# model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}")
# tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")

# train_dataset = load_dataset("csv", data_files=train_data_file, split="train")

# training_args = DPOConfig(
#     output_dir=f"{model_name}_DPO_0", loss_type="sigmoid", truncation_mode="keep_end"
# )
# trainer = DPOTrainer(
#     model=model,
#     args=training_args,
#     processing_class=tokenizer,
#     train_dataset=train_dataset,
# )
# trainer.train()
# trainer.save_model(exp_dir)
# artifact = wandb.Artifact("trained-model", type="model")
# artifact.add_dir(exp_dir)
# wandb.log_artifact(artifact)

from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

model_name = "Qwen2.5-Coder-0.5B-Instruct"
train_data_file = "/workspace/cs229/finetuning/datasets/20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8_trajectories_preference_train.csv"
test_data_file = "/workspace/cs229/finetuning/datasets/20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8_trajectories_preference_test.csv"
exp_dir = f"/workspace/cs229/finetuning/models/{model_name}_DPO_1"

load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
max_seq_length = int(40e3)
dtype = None

# model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}")
# tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"unsloth/{model_name}",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    chosen = examples["chosen"]
    rejected = examples["rejected"]
    score_chosen = examples["score_chosen"]
    score_rejected = examples["score_rejected"]
    chosen_eos = []
    rejected_eos = []
    for c, r in zip(chosen, rejected):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        c += EOS_TOKEN
        r += EOS_TOKEN
        chosen_eos.append(c)
        rejected_eos.append(r)
    return {
        "chosen": chosen_eos,
        "rejected": rejected_eos,
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
    }


pass

train_dataset = load_dataset("csv", data_files=train_data_file, split="train")
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
)
eval_dataset = load_dataset("csv", data_files=test_data_file, split="test")
eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched=True,
)

training_args = DPOConfig(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    num_train_epochs=1,
    learning_rate=5e-6,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    seed=42,
    output_dir=f"{model_name}_DPO_0",
    loss_type="sigmoid",
    truncation_mode="keep_end",
    max_length=max_seq_length,
    generate_during_eval=True,
    eval_strategy="steps",
    eval_steps=25,
)
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
trainer.save_model(exp_dir)
artifact = wandb.Artifact("trained-model", type="model")
artifact.add_dir(exp_dir)
wandb.log_artifact(artifact)
