import random
from dataclasses import dataclass

import datasets
from typing import Union, List, Tuple, Dict

import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm

class PointDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self.ir_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'qry': [datasets.Value("int32")],
                'psg': [datasets.Value("int32")],
                'label': datasets.Value("int32"),
            })
        )['train']
        self.total_len = len(self.ir_dataset)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        irdata = self.ir_dataset[item]
        encoded_qry = irdata['qry']
        encoded_psg = irdata['psg']
        label = irdata['label']
        encoding = self._tokenizer.encode_plus(
            encoded_qry,
            encoded_psg,
            truncation='only_second',
            max_length=self._max_seq_length,
            padding='max_length',
        )
        # return encoding
        return {
            "input_ids": np.array(encoding['input_ids']),
            "token_type_ids": np.array(encoding['token_type_ids']),
            "attention_mask": np.array(encoding['attention_mask']),
            "label": int(label)
        }
