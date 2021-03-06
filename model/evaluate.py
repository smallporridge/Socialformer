import numpy as np
import argparse
import gzip
import os
from tqdm import tqdm
import shutil

def average_precision(gt, pred):
	"""
	Computes the average precision.

	This function computes the average prescision at k between two lists of
	items.

	Parameters
	----------
	gt: set
			 A set of ground-truth elements (order doesn't matter)
	pred: list
				A list of predicted elements (order does matter)

	Returns
	-------
	score: double
			The average precision over the input lists
	"""

	if not gt:
		return 0.0

	score = 0.0
	num_hits = 0.0
	for i,p in enumerate(pred):
		if p in gt and p not in pred[:i]:
			num_hits += 1.0
			score += num_hits / (i + 1.0)

	return score / max(1.0, len(gt))


def NDCG(gt, pred, use_graded_scores=False):
	score = 0.0
	for rank, item in enumerate(pred):
		if item in gt:
			if use_graded_scores:
				grade = 1.0 / (gt.index(item) + 1)
			else:
				grade = 1.0
			score += grade / np.log2(rank + 2)

	norm = 0.0
	for rank in range(len(gt)):
		if use_graded_scores:
			grade = 1.0 / (rank + 1)
		else:
			grade = 1.0
		norm += grade / np.log2(rank + 2)
	return score / max(0.3, norm)


def metrics(gt, pred, metrics_map):
	'''
	Returns a numpy array containing metrics specified by metrics_map.
	gt: set
			A set of ground-truth elements (order doesn't matter)
	pred: list
			A list of predicted elements (order does matter)
	'''
	out = np.zeros((len(metrics_map),), np.float32)

	if ('MAP@20' in metrics_map):
		avg_precision = average_precision(gt=gt, pred=pred[:20])
		out[metrics_map.index('MAP@20')] = avg_precision

	if ('P@20' in metrics_map):
		intersec = len(gt & set(pred[:20]))
		out[metrics_map.index('P@20')] = intersec / max(1., float(len(pred[:20])))

	if 'MRR' in metrics_map:
		score = 0.0
		for rank, item in enumerate(pred):
			if item in gt:
				score = 1.0 / (rank + 1.0)
				break
		out[metrics_map.index('MRR')] = score

	if 'MRR@100' in metrics_map:
		score = 0.0
		for rank, item in enumerate(pred[:100]):
			if item in gt:
				score = 1.0 / (rank + 1.0)
				break

		out[metrics_map.index('MRR@100')] = score

	if 'MRR@10' in metrics_map:
		score = 0.0
		for rank, item in enumerate(pred[:10]):
			if item in gt:
				score = 1.0 / (rank + 1.0)
				break
		
		out[metrics_map.index('MRR@10')] = score


	if ('NDCG@20' in metrics_map):
		out[metrics_map.index('NDCG@20')] = NDCG(gt, pred[:20])
	
	if ('NDCG@10' in metrics_map):
		out[metrics_map.index('NDCG@10')] = NDCG(gt, pred[:10])

	if ('NDCG@100' in metrics_map):
		out[metrics_map.index('NDCG@100')] = NDCG(gt, pred[:100])
	return out

class evaluator:
	def __init__(self, qrels_path, score_path):

		self.METRICS_MAP = ['MRR@100', 'MRR@10', 'NDCG@100', 'NDCG@20', 'NDCG@10', 'MAP@20', 'P@20']

		qrels_dev = {}
		with open(qrels_path) as qf:
			for line in qf:
				qid, _, docid, _ = line.strip().split()
				if qid in qrels_dev:
					qrels_dev[qid].append(docid)
				else:
					qrels_dev[qid] = [docid]
		scores_dev = {}

		with open(score_path) as sf:
			for line in sf:
				qid, docid, score = line.strip().split()
				if qid in scores_dev:
					scores_dev[qid] += [(docid, float(score))]
				else:
					scores_dev[qid] = [(docid, float(score))]
		
		for qid in scores_dev:
			scores_dev[qid] = sorted(scores_dev[qid], key=lambda x: x[1], reverse=True)
		
		self.qrels_dev = qrels_dev
		self.scores_dev = scores_dev
	
	def evaluate(self):

		c_n = 0
		map_list = []
		p_list = []
		mrr_list = []
		ndcg_100_list = []
		ndcg_20_list = []
		ndcg_10_list = []

		mrr_100_list = []
		mrr_10_list = []
		for qid in self.scores_dev:
			if qid in self.qrels_dev:
				gold_set = set(self.qrels_dev[qid])
				y = [s[0] for s in self.scores_dev[qid]]
				# _map20, _p20, _mrr, _ndcg20, _mrr100, _mrr10 = metrics(gt=gold_set, pred=y, metrics_map=self.METRICS_MAP)
				_mrr100, _mrr10, _ndcg100, _ndcg20, _ndcg10, _map20, _p20 = metrics(gt=gold_set, pred=y, metrics_map=self.METRICS_MAP)

				mrr_100_list += [_mrr100]
				mrr_10_list += [_mrr10]      

				ndcg_100_list += [_ndcg100]
				ndcg_20_list += [_ndcg20]
				ndcg_10_list += [_ndcg10]

				map_list += [_map20]
				p_list += [_p20]
				
				
			else:
				c_n += 1
		
		# print([np.mean(map_list), np.mean(p_list), np.mean(mrr_list), np.mean(ndcg_list), np.mean(mrr_100_list), np.mean(mrr_10_list)])
		# print(f"map@20: {np.mean(map_list)}, p@20:{np.mean(p_list)}, ndcg@20:{np.mean(ndcg_list)}, mrr@100:{np.mean(mrr_100_list)}")
		return [np.mean(mrr_100_list), np.mean(mrr_10_list), np.mean(ndcg_100_list), np.mean(ndcg_20_list), np.mean(ndcg_10_list), np.mean(map_list), np.mean(p_list)]

class evaluator_trec:
	
	def __init__(self, qrels_path, score_path):
		self.qrels_path = qrels_path
		self.score_path = score_path
	
	def evaluate(self):

		import pytrec_eval
		import json

		qrel = {}

		with open(self.qrels_path, 'r') as qf:
			for line in qf:
				qid, _, pid, rel = line.strip().split()
				if qid not in qrel:
					qrel[qid] = {pid: int(rel)}
				else:
					qrel[qid][pid] = int(rel)
		run = {}
		with open(self.score_path, 'r') as sf:
			for line in sf:
				# qid, _, pid, rank, score, _ =  line.strip().split()
				qid, pid, score = line.strip().split()
				if qid not in run:
					run[qid] = {pid: float(score)}
				else:
					run[qid][pid]  = float(score)
		qids = []
		for qid in run:
			qids += [qid]


		evaluator = pytrec_eval.RelevanceEvaluator(
			qrel, {'map_cut', 'ndcg_cut', 'P'})

		# print(json.dumps(evaluator.evaluate(run), indent=1))
		result = evaluator.evaluate(run)
		# print(result[qids[0]])

		ndcg_100_list = []
		ndcg_10_list = []
		ndcg_20_list = []
		p_20_list = []

		for qid in qids:
			ndcg_100_list += [result[qid]['ndcg_cut_100']]
			ndcg_10_list += [result[qid]['ndcg_cut_10']]
			ndcg_20_list += [result[qid]['ndcg_cut_20']]
			p_20_list += [result[qid]['P_20']]
		
		ndcg100 = sum(ndcg_100_list) / len(ndcg_100_list)
		ndcg10 = sum(ndcg_10_list) / len(ndcg_10_list)
		ndcg20 = sum(ndcg_20_list) / len(ndcg_20_list)
		p20 = sum(p_20_list) / len(p_20_list)


		print("ndcg_100:", ndcg100)
		print("ndcg_10:", ndcg10)
		print("ndcg_20:", ndcg20)
		print("p_20:", p20)

		return ndcg100, ndcg10, ndcg20, p20
