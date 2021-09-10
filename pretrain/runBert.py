import argparse
import torch
from torch.utils.data import DataLoader
from BertForPretrain import BertForPretrain
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from pretrain_dataset import PretrainDataset
from tqdm import tqdm
import os
from utils import *
# 全局的参数
device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save model.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--train_file", type=str)
parser.add_argument("--bert_model", type=str)
parser.add_argument("--dataset_script_dir", type=str, help="The path to save log.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path to save log.")
args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
print(args)
logger = open(args.log_path, "a")
logger.write("\n")
tokenizer = BertTokenizer.from_pretrained(args.bert_model)

def load_data():
    train_dir = args.train_file
    fns = [os.path.join(train_dir, fn) for fn in os.listdir(train_dir)]
    train_data = fns
    return train_data

def train_model(train_data):
    bert_model = BertModel.from_pretrained(args.bert_model)
    model = BertForPretrain(bert_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data) # 开始训练

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    loss = model.forward(train_data)
    return loss

def fit(model, X_train):
    train_dataset = PretrainDataset(X_train, 512, tokenizer, args.dataset_script_dir, args.dataset_cache_dir) # 构建训练集
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data) # 过模型, 取loss
            loss = loss.mean()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step() # 更新模型参数
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())
            avg_loss += loss.item()
        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
    logger.close()
    save_model(model, tokenizer, args.save_path)

if __name__ == '__main__':
    train_data = load_data() # 加载数据集
    set_seed() # 控制各种随机种子
    train_model(train_data) # 开始预训练
