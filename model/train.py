import time
import argparse
import pickle
import random
import numpy as np
import torch
import logging
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import BertForSearch
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
# from Trec_Metrics import Metrics
# from pair_dataset import PairDataset
from point_dataset_all import PointDataset
from list_dataset_all import ListDataset
from tqdm import tqdm
import os
from evaluate import evaluator, evaluator_trec

parser = argparse.ArgumentParser()
parser.add_argument("--is_training",action="store_true")
parser.add_argument("--per_gpu_batch_size",
                    default=25,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=64,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--task",
                    default="msmarco-doc",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=2,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_docs",
                    default=8,
                    type=int,
                    help="Max number of documents per query.")
parser.add_argument("--max_groups",
                    default=16,
                    type=int,
                    help="Max number of subgraphs.")
parser.add_argument("--max_psglen",
                    default=128,
                    type=int,
                    help="Max number of passage length.")
parser.add_argument("--sub_graph",
                    default='sub_graph1',
                    type=str,
                    help="graph partition method.")
parser.add_argument("--aggregator",
                    default='max',
                    type=str,
                    help="pooling method.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--msmarco_score_file_path",
                    type=str,
                    help="The path to save score file.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")

parser.add_argument("--train_file",
                    type=str)
parser.add_argument("--dev_file",
                    type=str)
parser.add_argument("--dev_id_file",
                    type=str)
parser.add_argument("--bert_model",
                    type=str)
parser.add_argument("--dataset_script_dir",
                    type=str,
                    help="-.")
parser.add_argument("--dataset_cache_dir",
                    type=str,
                    help="-.")
parser.add_argument("--msmarco_dev_qrel_path",
                    type=str,
                    help="The path of relevance file.")

args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
result_path = "./output/" + args.task + "/"
# args.save_path += BertForSearch.__name__ + "." +  args.task
score_file_prefix = result_path + BertForSearch.__name__ + "." + args.task
# args.log_path += BertForSearch.__name__ + "." + args.task + ".log"
# args.msmarco_score_file_path = score_file_prefix + "." +  args.msmarco_score_file_path

logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\n")
logger.write(str(args)+'\n')

train_dir = args.train_file
fns = [os.path.join(train_dir, fn) for fn in os.listdir(train_dir)]
train_data = fns

dev_data = args.dev_file
tokenizer = BertTokenizer.from_pretrained(args.bert_model)


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

def train_model():
    bert_model = BertModel.from_pretrained(args.bert_model)
    model = BertForSearch(bert_model, args.max_docs, args.max_groups + 16, args.max_psglen+5)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, dev_data)

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)

    loss = model.forward(train_data, is_training=True, pooling=args.aggregator)
    return loss

def fit(model, X_train, X_test):
    train_dataset = ListDataset(X_train, args.sub_graph, args.max_groups, args.max_psglen, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            
            loss = train_step(model, training_data)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())
            if i == 1:
                print("start training")
            if i % 100 == 0:
                print(i, loss.item())

            if i > 0 and i == (one_epoch_step//5) == 0: # test the model per 20% steps
                best_result = evaluate(model, X_test, best_result)
                model.train()
            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        #best_result = evaluate(model, X_test, X_test_new, best_result)
    logger.close()

def evaluate(model, X_test, best_result, is_test=False):
    y_pred = predict(model, X_test)
    # print(y_pred)

    qid_pid_list = []
    with open(args.dev_id_file, 'r') as dif:
        for line in dif:
            qid, docid = line.strip().split()
            qid_pid_list.append([qid, docid])
    
    # print(len(y_pred))
    # print(len(qid_pid_list))

    fw = open(args.msmarco_score_file_path, 'w')
    for i, (qd, y_pred) in enumerate(zip(qid_pid_list, y_pred)):
        qid, pid = qd
        fw.write(qid + "\t" + pid + "\t" + str(y_pred) + "\n")
    fw.close()

    if args.task == "msmarco":
        myevaluator = evaluator(args.msmarco_dev_qrel_path, args.msmarco_score_file_path)
    elif args.task == "trecdl":
        myevaluator = evaluator_trec(args.msmarco_dev_qrel_path, args.msmarco_score_file_path)
    
    result = myevaluator.evaluate()
    

    if not is_test:
        if args.task == "msmarco" and result[-2] > best_result[-2]:
            best_result = result
            print("[best result]", result)
            _mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = result
            
            tqdm.write(f"[best result][msmarco][{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}")
            logger.write(f"[best result][msmarco][{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}\n")
            
            logger.flush()
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.save_path)

        if args.task == "msmarco" and result[-2] <= best_result[-2]:
            print("[normal result]", result)
            _mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = result
            logger.write(f"[normal result][msmarco][{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}\n")
            logger.flush()
         
    if is_test and args.task == "msmarco":
        _mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = result
        tqdm.write(f"[{args.id}] mrr@100:{_mrr100}, mrr@10:{_mrr10}, ndcg@100:{_ndcg100}, ndcg@20:{_ndcg20}, ndcg@10:{_ndcg10}, map@20:{_map20}, p@20:{_p20}")
    
    if is_test and args.task == "trecdl":
        _ndcg100, _ndcg10, _ndcg20, _p20 = result
        logger.write(f"[normal result][trecdl][{args.id}] _ndcg100:{_ndcg100}, _ndcg10:{_ndcg10}, _ndcg20:{_ndcg20}, _p20:{_p20}\n")
        logger.flush()

    return best_result

def predict(model, X_test):
    model.eval()
    test_loss = []
    test_dataset = PointDataset(X_test, args.sub_graph, args.max_groups, args.max_psglen, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    # y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data, is_training=False, pooling=args.aggregator) # bs
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            # y_tmp_label = test_data["labels"].data.cpu().numpy().reshape(-1)
            # y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    # y_label = np.concatenate(y_label, axis=0).tolist()
    return y_pred

def test_model():
    bert_model = BertModel.from_pretrained(args.bert_model)
    model = BertForSearch(bert_model, args.max_docs, args.max_groups + 16, args.max_psglen+5)
    #model = BertForSearch(bert_model, 2 * int(args.max_doc_len/args.window_size), args.window_size+5)
    # model.bert_model.resize_token_embeddings(model.bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate(model, dev_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)

if __name__ == '__main__':
    set_seed()
    if args.is_training:
        train_model()
    else:
        test_model()
