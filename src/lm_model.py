from transformers import (
    AutoTokenizer, 
    BertConfig, 
    BertForMaskedLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments, 
    pipeline
)

from datasets import Dataset
import evaluate
import torch
import numpy as np
import pandas as pd
import itertools

class AfrikaLM:
    def __init__(self, data_paths, num_hiddens, num_attentions, base_model = 'bert-base-cased', vocab_size = None) -> None:
        self.block_size = 128 # block size for training data
        self.mlm_probability = 0.15 # probability of mask token

        self.data_paths = data_paths # Could be a list of datadir for using multiple datasets
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_attentions = num_attentions
        
        self.load_dataset() # adds self.dataset
        self.build_and_train_tokenizer() # adds self.tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability)

        self.create_model() # adds self.model
    
    def create_model(self):
        config = BertConfig.from_pretrained(self.base_model, 
                                            num_hidden_layers = self.num_hiddens, 
                                            num_attention_heads = self.num_attentions, 
                                            vocab_size=self.tokenizer.vocab_size)
        self.model = BertForMaskedLM(config)

    def build_and_train_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer = tokenizer.train_new_from_iterator(self.dataset['train']['text'], tokenizer.vocab_size if self.vocab_size is None else self.vocab_size)
        self.tokenizer = tokenizer
    
    def load_dataset(self):
        ds = Dataset.from_text(self.data_paths)
        ds = ds.shuffle(42)
        ds = ds.train_test_split(test_size=0.2)
        self.dataset = ds 
    
    def get_lm_dataset(self):
        """ Tokenize the dataset,
        Concantenate all text together,
        Split the concatenated text in chunks of `block_size``
        """

        def preprocess_function(examples):
            return self.tokenizer([" ".join(x) for x in examples["text"]])

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= self.block_size:
                total_length = (total_length // self.block_size) * self.block_size
            # Split by chunks of block_size.
            result = {
                k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_ds = self.dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=self.dataset['train'].column_names
        )
        
        lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)

        return lm_dataset
    
    def evaluate(self):
        """Return perplexity"""
        eval_results = self.trainer.evaluate()
        return torch.exp(torch.tensor(eval_results['eval_loss']))

    def train_model(self, output_dir, learning_rate=2e-5, num_train_epochs=2, weight_decay=0.01, batch_size=16, validate_on_test = False, push_to_hub=True):
        
        lm_dataset = self.get_lm_dataset()

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to='tensorboard',
            save_total_limit=1,
            push_to_hub=push_to_hub
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=lm_dataset['train'],
            eval_dataset=lm_dataset['test'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        self.trainer.train()

    def predict(self, text, topk=3):
        mask_filler = pipeline("fill-mask", model=self.model.to("cpu"), tokenizer=self.tokenizer)
        ret = mask_filler(text, top_k=topk)

        return ret                    