    
import random, os, torch
import numpy as np
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    # torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    model_to_save.bert_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


