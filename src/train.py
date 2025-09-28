import torch
import pandas as pd
from pathlib import Path
import yaml
import argparse
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)
from tokenizers.normalizers import Lowercase, Strip, StripAccents, NFD
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    LlamaConfig)
from datasets import load_dataset

def train_tokenizer(config, output_dir):
    """Train a new tokenizer on the training data."""
    tokenizer = Tokenizer(models.BPE())
    normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip()])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=config['tokenizer']['vocab_size'],
        special_tokens=["<|endoftext|>", "<pad>"]
    )

    # Ensure training_files are in a list format
    training_files = [config['data']['train_files']]
    print(f"Training tokenizer on: {training_files}")
    tokenizer.train(files=training_files, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    # Save the tokenizer
    tokenizer_path = output_dir / 'tokenizer'
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
    )
    wrapped_tokenizer.save_pretrained(str(tokenizer_path))
    return wrapped_tokenizer

def prepare_dataset(tokenizer, config):
    """Prepare the dataset for training."""
    # Load datasets
    raw_datasets = load_dataset(
        'text',
        data_files={
            'train': config['data']['train_files'],
            'validation': config['data']['eval_files']
        }
    )

    context_length = config['model']['context_length']

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets

def train_model(config_path):
    """Train a BabyLM model using the provided configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up output directory
    output_dir = Path(f"models/{config['model_name']}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train tokenizer
    tokenizer = train_tokenizer(config, output_dir)

    # Prepare dataset
    tokenized_datasets = prepare_dataset(tokenizer, config)

    # Initialize model
    model_config = LlamaConfig(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        intermediate_size=config['model']['intermediate_size'],
        num_attention_heads=config['model']['num_attention_heads'],
        max_position_embeddings=config['model']['max_position_embeddings'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Set random seed
    set_seed(config['training']['seed'])
    model = LlamaForCausalLM(model_config)
    print(f'Model {config["model_name"]} num parameters = {model.num_parameters():,}')

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_strategy="epoch",
        eval_strategy="epoch",
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        report_to="none",  # Disable wandb logging
        use_mps_device=True,  # Using Apple Silicon
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    trainer.train()

    # Save training logs
    log_history_df = pd.DataFrame(trainer.state.log_history)
    log_history_df.to_csv(str(output_dir / 'training_logs.csv'))

    # Save final model
    trainer.save_model(str(output_dir / 'final'))
    print(f"Model saved to {output_dir / 'final'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BabyLM model.")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to the model configuration YAML file.")
    args = parser.parse_args()
    train_model(args.config)