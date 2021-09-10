
import datasets
import torch
from torch.utils.data import Dataset
import numpy as np
class PretrainDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files = self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'input_ids': [datasets.Value("int32")],
                'token_type_ids': [datasets.Value("int32")],
                'attention_mask': [datasets.Value("int32")],
                'mlm_labels': [datasets.Value("int32")],
                'input_terms': [datasets.Value("string")],
                'label': datasets.Value("int32"),
            })
        )['train']
        self.total_len = len(self.nlp_dataset)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        return {
            "input_ids": np.array(data['input_ids']),
            "token_type_ids": np.array(data['token_type_ids']),
            "attention_mask": np.array(data['attention_mask']),
            "mlm_labels": np.array(data['mlm_labels']),
            "label": int(data['label'])
        }
