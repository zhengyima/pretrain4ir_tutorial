import argparse
import numpy as np
import torch
import logging
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from BertForSearch import BertForSearch
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from point_dataset import PointDataset
from tqdm import tqdm
import os
from evaluate import evaluator
from utils import *
device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("--is_training",action="store_true")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size", default=64, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save model.")
parser.add_argument("--msmarco_score_file_path", type=str, help="The path to save model.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--train_file", type=str)
parser.add_argument("--dev_file", type=str)
parser.add_argument("--dev_id_file", type=str)
parser.add_argument("--bert_model", type=str)
parser.add_argument("--dataset_script_dir", type=str, help="The path to save log.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path to save log.")
parser.add_argument("--msmarco_dev_qrel_path", type=str, help="The path to save log.")
parser.add_argument("--id", type=str, default='default', help="The path to save log.")
args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
print(args)
logger = open(args.log_path, "a")
logger.write("\n")
tokenizer = BertTokenizer.from_pretrained(args.bert_model)

def load_data():
    train_dir = args.train_file
    fns = [os.path.join(train_dir, fn) for fn in os.listdir(train_dir)]
    train_data = fns
    dev_data = args.dev_file
    return train_data, dev_data

def train_model(train_data, dev_data):
    bert_model = BertModel.from_pretrained(args.bert_model)
    model = BertForSearch(bert_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, dev_data)

def train_step(model, train_data, loss):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    y_pred = model.forward(train_data)
    batch_y = train_data["label"]
    loss = loss(y_pred.view(-1, 2), batch_y.view(-1))
    return loss

def fit(model, X_train, X_test):
    train_dataset = PointDataset(X_train, 512, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    fct_loss = torch.nn.CrossEntropyLoss()
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data, fct_loss)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())
            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        best_result = evaluate(model, X_test, best_result)
    logger.close()

def evaluate(model, X_test, best_result, is_test=False):
    y_pred = predict(model, X_test)
    qid_pid_list = []
    with open(args.dev_id_file, 'r') as dif:
        for line in dif:
            qid, docid = line.strip().split()
            qid_pid_list.append([qid, docid])

    fw = open(args.msmarco_score_file_path, 'w')
    for i, (qd, y_pred) in enumerate(zip(qid_pid_list, y_pred)):
        qid, pid = qd
        fw.write(qid + "\t" + pid + "\t" + str(y_pred) + "\n")
    fw.close()
    myevaluator = evaluator(args.msmarco_dev_qrel_path, args.msmarco_score_file_path)
    result = myevaluator.evaluate()
    
    if not is_test:
        if result[-2] > best_result[-2]:
            best_result = result
            print("[best result]", result)
            _mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = result
            tqdm.write(f"[best result][msmarco][{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}")
            logger.write(f"[best result][msmarco][{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}\n")
            logger.flush()
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.save_path)

        if result[-2] <= best_result[-2]:
            print("[normal result]", result)
            _mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = result
            logger.write(f"[normal result][msmarco][{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}\n")
            logger.flush()

    _mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = result
    tqdm.write(f"[{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}")
    return best_result

def predict(model, X_test):
    model.eval()
    test_loss = []
    test_dataset = PointDataset(X_test, 512, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data) # bs, 2
            y_pred_test = F.softmax(y_pred_test, dim=1)[:,1] # bs
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    return y_pred

def test_model(dev_data):
    bert_model = BertModel.from_pretrained(args.bert_model)

    model = BertForSearch(bert_model)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate(model, dev_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)

if __name__ == '__main__':
    train_data, dev_data = load_data()
    set_seed()
    if args.is_training:
        train_model(train_data, dev_data)
    else:
        test_model(dev_data)
