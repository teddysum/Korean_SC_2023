
import json
import pandas as pd

from datasets import Dataset
from torch.utils.data import DataLoader
from tokenizers.processors import TemplateProcessing

def StoryDataLoader(fname, tokenizer, batch_size, max_length, mode="train"):
    """
    Build Data Loader

    """

    dataset = Dataset.from_json(fname, mode)

    if not tokenizer.cls_token:
        tokenizer.cls_token = tokenizer.bos_token
    if not tokenizer.sep_token:
        tokenizer.sep_token = tokenizer.eos_token

    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.cls_token} $0 {tokenizer.sep_token}",
        pair=f"{tokenizer.cls_token} $A {tokenizer.sep_token} $B:1 {tokenizer.sep_token}:1",
        special_tokens=[(tokenizer.cls_token, tokenizer.cls_token_id), (tokenizer.sep_token, tokenizer.sep_token_id)],
    )

    def preprocess_function(examples):
        processed = {}
        tokenizer_input = tokenizer(
            examples["input"]["sentence1"],
            examples["input"]["sentence3"],
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        processed["input_ids"] = tokenizer_input["input_ids"],
        processed["attention_mask"] = tokenizer_input["attention_mask"]

        if mode == "train":
            tokenizer_output = tokenizer(
                examples["output"],
                padding="max_length",
                max_length=max_length,
                truncation=True
            )
            processed["decoder_input_ids"] = tokenizer_output["input_ids"]
            processed["decoder_attention_mask"] = tokenizer_output["attention_mask"]

        return processed

    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names
    ).with_format("torch")
    dataloader = DataLoader(dataset, shuffle=(True if mode=="train" else False), batch_size=batch_size)

    return dataloader


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
