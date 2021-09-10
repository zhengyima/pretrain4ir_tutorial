import os, argparse, random, json
from transformers import BertTokenizer
from tqdm import tqdm
# 该脚本实现了, 从MSMARCO Document Ranking数据集的文档中, 每个文档生成一个正例一个负例. 
# 正例负例的query都是随机生成长度在1-5之间的词作为query.
# 会对Document部分做Mask, 从而支持MLM预训练任务. 见create_masks_for_sequence函数
parser = argparse.ArgumentParser()
parser.add_argument("--msmarco_docs_path",type=str,help='refer to https://github.com/microsoft/MSMARCO-Document-Ranking')
parser.add_argument("--bert_model",type=str,help='bert model checkpoint dir, download from huggingface')
parser.add_argument("--output_file",type=str, 'output path of the generated datas')
parser.add_argument("--docs_limit",type=int,default=10000000000)
parser.add_argument("--mlm_prob",type=float,default=0.15)
args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(args.bert_model)
MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")

def get_query_from_fulltext(fulltext):
    tokens = tokenizer.tokenize(fulltext)
    rand_q_len = random.randint(1,5)
    rand_indexs = random.sample(range(1, len(tokens)), rand_q_len)
    query_tokens = [tokens[idx] for idx in rand_indexs]
    query_token_str = " ".join(query_tokens)
    return query_token_str

def create_masks_for_sequence(input_ids):
    labels = [-100 for i in range(len(input_ids))]
    for i, input_id in enumerate(input_ids):
        if i == 0:
            continue
        if random.random() < args.mlm_prob:
            labels[i] = input_ids[i]
            input_ids[i] = MASK_TOKEN_ID
    return input_ids, labels

datas = []
with open(args.msmarco_docs_path) as fin:
    for i, line in tqdm(enumerate(fin), desc='processing data'):
        if i >= args.docs_limit: # for demo speed 
            break
        cols = line.split("\t")
        if len(cols) != 4:
            continue
        docid, url, title, body = cols 
        fulltext = url + " " + title + " " + body
        pos_query = get_query_from_fulltext(fulltext)
        neg_query = get_query_from_fulltext(fulltext)
        for j, query in enumerate([pos_query, neg_query]):
            if j == 0:
                label = 1
            else:
                label = 0
            encoded = tokenizer.encode_plus(query, fulltext, add_special_tokens=True, truncation='only_second', max_length=512, padding='max_length')
            input_ids = encoded['input_ids']
            token_type_ids = encoded['token_type_ids']
            attention_mask = encoded['attention_mask']

            input_ids, mlm_labels = create_masks_for_sequence(input_ids)
            data = {
                "input_ids": input_ids, 
                "token_type_ids": token_type_ids, 
                "attention_mask":attention_mask, 
                "mlm_labels":mlm_labels, 
                "input_terms": tokenizer.convert_ids_to_tokens(input_ids),
                "label":label
                }
            datas += [data]
        
with open(args.output_file, 'w') as fout:
    for d in tqdm(datas, desc='writing data...'):
        fout.write(json.dumps(d, ensure_ascii=False) + "\n")
