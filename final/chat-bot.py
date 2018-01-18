#!/usr/bin/env python
#coding=utf-8
# author: xu3352
# desc: str output test in python3

import numpy as np
import sys
import pandas as pd
import argparse
import jieba
import jieba.analyse
from gensim.models import Word2Vec
import gensim, logging
import pickle as pkl
import time
import copy
import random
import multiprocessing

parser = argparse.ArgumentParser(description='chat-bot')
parser.add_argument('--action', choices=['train', 'test', 'see'])
parser.add_argument('--training_data_path')
parser.add_argument('--testing_data_path')
parser.add_argument('--jieba_dic_path')
parser.add_argument('--w2v_model_save_path')
parser.add_argument('--w2v_model_load_path')
parser.add_argument('--w2v_min_count', type=int, default=0)
parser.add_argument('--w2v_load_option', type=bool)
parser.add_argument('--w2v_train_cutall', type=bool, default=False)
parser.add_argument('--w2v_voca_size', type=int, default=None)
parser.add_argument('--w2v_vec_dim', type=int)
parser.add_argument('--iter', type=int)
parser.add_argument('--search', type=bool)
parser.add_argument('--sg', type=bool)
args = parser.parse_args()

args.search = False

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
jieba.initialize()
jieba.set_dictionary(args.jieba_dic_path)
stopwordset = set()

if args.search:
	args.w2v_model_save_path = './word2vec_model/w2v_search_'
	args.w2v_model_load_path = './word2vec_model/w2v_search_'
else:
	if (args.w2v_train_cutall):
		args.w2v_model_save_path = './word2vec_model/w2v_cutall_'
		args.w2v_model_load_path = './word2vec_model/w2v_cutall_'
	else:
		args.w2v_model_save_path = './word2vec_model/w2v_nocutall_'
		args.w2v_model_load_path = './word2vec_model/w2v_nocutall_'

#set stop words
with open('./training_data/stopwords.txt','r',encoding='utf-8') as sw:
		for line in sw:
			stopwordset.add(line.strip('\n'))
#args.train




def train_w2v(sentences, dim, model_save_path='', model_load_path='', load=False, save=False):
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	print("cut: ", args.w2v_train_cutall)
	if load == True:
		print("load model...")
		model = Word2Vec.load(model_load_path)
		
		#total_examples=model.corpus_count
		model.train(sentences,total_examples=model.corpus_count, epochs=model.iter)
		#print(model.similar("床", "床上"))
		#model.save(model_save_path)

	else :
		model = Word2Vec(sentences, size = dim, min_count=args.w2v_min_count)
		#print(model.most_similar("床", "床上"))

		#print(model.most_similar("床"))
		#model.save(model_save_path)
	if save:
		print("saving to ", model_save_path)
		model.save(model_save_path)
	return model


def average_sentence(cutted_sentence, model, model_dim):
	emb_cnt = 0.
	#model = Word2Vec.load(args.word2vec_model_load_path)
	avg_dlg_emb = np.zeros((model_dim,))

	#print('cutting...')
	#cutted_sentence = jieba.lcut(sentence)

	#print('averaging..')
	for word in cutted_sentence:
		if word in model.wv.vocab:
			#print(word)
			avg_dlg_emb += model.wv[word]
			emb_cnt += 1.

	if emb_cnt != 0:
		avg_dlg_emb = avg_dlg_emb / emb_cnt

	#print(sentence, " aver:", avg_dlg_emb)

	#print(type(avg_dlg_emb))
	return avg_dlg_emb

def sim_sentences(sentence_vec_1, sentence_vec_2):
	zero = np.zeros(len(sentence_vec_1))
	
	if np.equal(zero, sentence_vec_1)[0] or np.equal(zero, sentence_vec_2)[0]:
		return 0
	else:
		sim = (np.dot(sentence_vec_1, sentence_vec_2) / np.linalg.norm(sentence_vec_1) )/ np.linalg.norm(sentence_vec_2)
	return sim

def split_testing_data(sentences):
	processed_data = []
	for i in range(len(sentences)):
		sentences[i] = sentences[i].split('\t')
	return sentences



def main():
	print("action: ", args.action)

	if args.action == 'train':
		if args.search:
			cutted_sentence_path = './cutted_data/cutted_search_data.pkl'
		else :
			if args.w2v_train_cutall:
				print("cutall!")
				cutted_sentence_path = './cutted_data/cutted_cutall_data.pkl'
			else:
				print("no cutall!")
				cutted_sentence_path = './cutted_data/cutted_nocutall_data.pkl'

		
		cutted_sentences = pkl.load(open(cutted_sentence_path, 'rb'))
		print("training path: ", cutted_sentence_path)
		print(cutted_sentences[0:10])

		for idx in range(1, 8):
			print('training for dim = ', str(args.w2v_vec_dim*idx))
			model_paths_add_dim = args.w2v_model_save_path+ "_sg" + str(args.sg) + "_" + "iter_"+ str(args.iter) + "_" + str(args.w2v_vec_dim*idx)
			cutted_sentence_path = ''
			model = Word2Vec(cutted_sentences,sg=args.sg,  iter = args.iter, window=5, size = int(args.w2v_vec_dim*idx), min_count=args.w2v_min_count, workers=multiprocessing.cpu_count())
			print("vocab size is", len(model.wv.vocab))
			print("save model to ", model_paths_add_dim)
			model.save(model_paths_add_dim)
	

	if args.action == 'see':


		df = pd.DataFrame(pd.read_csv("./testing_data.csv",encoding='utf-8'))
		test_id = df.values[:, 0]
		test_question = split_testing_data(df.values[:, 1])
		test_ans = split_testing_data(df.values[:, 2])

		

		model_paths = []
		model_dim = []
		for idx_path in range(2, 5):
			model_paths.append(args.w2v_model_load_path+str(64*idx_path))
			model_dim.append(64*idx_path)

		print(model_paths)

		cutted_ques = []
		cutted_ans = []
		
		cutted_ans_path = ''
		cutted_que_path = ''
		if args.w2v_train_cutall:
			print("cutall!")
			cutted_que_path = './cutted_ques_cutall.pkl'
			cutted_ans_path = './cutted_ans_cutall.pkl'
			pre_path = './prediction/prediction_cutall_{}.csv'
		else:
			print("no cutall!")
			cutted_que_path = './cutted_ques_nocutall.pkl'
			cutted_ans_path = './cutted_ans_nocutall.pkl'
			pre_path = './prediction/prediction_nocutall_{}.csv'

		
		cutted_ques = pkl.load(open(cutted_que_path, 'rb'))
		cutted_ans = pkl.load(open(cutted_ans_path, 'rb'))

		print("data amount is", len(cutted_ques))

		for idx_path, path in enumerate(model_paths):
			
			model = Word2Vec.load(path)
			print("vocab size is ", len(model.wv.vocab))
			#print(model.most_similar("怎麼辦"))
			print("testing for dim = ", model_dim[idx_path])
		
			option = []

			start = time.time()
			print("testing...")
			random.seed(time.time())
			rand_nb = random.randrange(5, len(cutted_ques))
			print("randnum: ", rand_nb)
			random_range = range(rand_nb-5, rand_nb)

			for idx_ques in random_range:
				print(test_question[idx_ques])
				max_sim = 0
				tmp_idx = 0

				tmp_ls = []
				
				for idx_ans, ans in enumerate(cutted_ans[idx_ques]):
					#tags_ans = jieba.analyse.extract_tags(str(ans))
					#print(cutted_ans)
					sim = sim_sentences(average_sentence(ans, model, model_dim[idx_path]), average_sentence(cutted_ques[idx_ques], model, model_dim[idx_path]))
					#sim = model.n_similarity(cutted_ques[idx_ques], ans) 
					tmp_ls.append((ans, sim))

				sorted_ls = sorted(tmp_ls, key=lambda tup: tup[1], reverse=True)

				for tup in sorted_ls:
					print(tup[0]," : ", tup[1])
				print()
			print()

				
	if args.action == 'test':

		model_paths = []
		model_dim = []
		for idx_path in range(2, 5):
			model_paths.append(args.w2v_model_load_path+str(64*idx_path))
			model_dim.append(64*idx_path)

		print(model_paths)

		cutted_ques = []
		cutted_ans = []
		
		cutted_ans_path = ''
		cutted_que_path = ''
		if args.w2v_train_cutall:
			print("cutall!")
			cutted_que_path = './cutted_ques_cutall.pkl'
			cutted_ans_path = './cutted_ans_cutall.pkl'
			pre_path = './prediction/prediction_cutall_{}.csv'
		else:
			print("no cutall!")
			cutted_que_path = './cutted_ques_nocutall.pkl'
			cutted_ans_path = './cutted_ans_nocutall.pkl'
			pre_path = './prediction/prediction_nocutall_{}.csv'

		print("load ques from: ", cutted_que_path)
		print("load ans from: ", cutted_ans_path)
		cutted_ques = pkl.load(open(cutted_que_path, 'rb'))
		print(cutted_ques[0])
		cutted_ans = pkl.load(open(cutted_ans_path, 'rb'))
		print("data amount is", len(cutted_ques))

		for idx_path, path in enumerate(model_paths):
			model = Word2Vec.load(path)
			print("vocab size is ", len(model.wv.vocab))
			#print(model.most_similar("怎麼辦"))
			print("testing for dim = ", model_dim[idx_path])
		
			option = []
			start = time.time()
			print("testing...")
			for idx_ques in range(len(cutted_ques)):
				if idx_ques%100 == 0 and idx_ques!= 0 :
					end = time.time()
					print(int(idx_ques/len(cutted_ques)*100), "% time used: ", int(end - start), "s", end ='\r')
					#time.sleep(1)
				max_sim = 0
				tmp_idx = 0
				
				for idx_ans, ans in enumerate(cutted_ans[idx_ques]):
					#tags_ans = jieba.analyse.extract_tags(str(ans))
					#print(cutted_ans)
					sim = sim_sentences(average_sentence(ans, model, model_dim[idx_path]), average_sentence(cutted_ques[idx_ques], model, model_dim[idx_path]))
					#print(type(sim))
					if sim > max_sim:
						tmp_idx = idx_ans
						max_sim = sim
				#print(tmp_idx)
				#print(max_sim)
				option.append(tmp_idx)
				#print(option)
				#print(test_ans[idx][tmp_idx], " : ", max_sim)
			#print(option)
			print()
			data_id = pkl.load(open('./test_id', 'rb'))
			df_output = pd.DataFrame(option, columns = ["ans"], index = data_id)
			df_output.to_csv(pre_path.format(str(64*(idx_path+2))), index_label = 'id')
			
if __name__ == '__main__':
	main()