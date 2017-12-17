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
args = parser.parse_args()



sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
jieba.initialize()
jieba.set_dictionary(args.jieba_dic_path)
strip_ls = ['「', '」', '\n', 'A:', 'B:', " ", "C:", '...']
#args.train

def data_preprocess(data_path):
	train = []

	for line in open(data_path).readlines():
		for char in strip_ls:
			line = line.replace(char, '')
		train.append(line)
	return train


def jieba_cut(sentences):

	cutted_sentences = []
	for sentence in sentences:
		#cutted_sentence = jieba.lcut(sentence, cut_all=args.w2v_train_cutall)
		cutted_sentence = jieba.lcut(sentence, cut_all=args.w2v_train_cutall)
		#cutted_sentence = list(jieba.cut_for_search(sentence))
		cutted_sentences.append(cutted_sentence)
	return cutted_sentences

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
	avg_dlg_emb = avg_dlg_emb / emb_cnt

	#print(sentence, " aver:", avg_dlg_emb)

	return avg_dlg_emb

def sim_sentences(sentence_vec_1, sentence_vec_2):
	sim = np.dot(sentence_vec_1, sentence_vec_2) / np.linalg.norm(sentence_vec_1) / np.linalg.norm(sentence_vec_2)
	return sim

def preprocess_testing_data(sentences):
	processed_data = []
	for i in range(len(sentences)):
		for char in strip_ls:
			sentences[i] = sentences[i].replace(char, '')
		sentences[i] = sentences[i].split('\t')
	return sentences




def main():
	print("action: ", args.action)
	if args.action == 'train':
		cutted_sentences = []
		for idx in range(1, 9):
			print('training for dim = ', str(args.w2v_vec_dim*idx))
			model_paths_add_dim = args.w2v_model_save_path + str(args.w2v_vec_dim*idx)
			#print(cutted_sentences[:10])
			#print("load: ", args.w2v_load_option)

			cutted_sentence_path = ''
			if args.w2v_train_cutall:
				cutted_sentence_path = 'cutted_cutall_data.pkl'
			else:
				cutted_sentence_path = 'cutted_nocutall_data.pkl'

			
			if (idx == 1):
				for path_idx in range(1, 6):
					path = './training_data/{}_train.txt'.format(str(path_idx))
					print("training from :", path)
					processed_data = data_preprocess(path)
					print('cutting...')
					print("first iter...")
					cutted_sentences = cutted_sentences + jieba_cut(processed_data)
					
					if path_idx == 5:
						print("dump cutted into", cutted_sentence_path)
						pkl.dump(cutted_sentences, open(cutted_sentence_path, 'wb'))
				
			#print(cutted_sentences)
			model = Word2Vec(cutted_sentences, size = int(args.w2v_vec_dim*idx), min_count=args.w2v_min_count)
			print("vocab size is", len(model.wv.vocab))
			print("save model to ", model_paths_add_dim)
			model.save(model_paths_add_dim)
			print()
	if args.action == 'see':
		model = Word2Vec.load(args.word2vec_model_load_path)
		#print(model.most_similar("怎麼辦"))
		
		
		df = pd.DataFrame(pd.read_csv(args.testing_data_path,encoding='utf-8'))
		test_id = df.values[:, 0]
		test_question = preprocess_testing_data(df.values[:10, 1])
		test_ans = preprocess_testing_data(df.values[:, 2])
		idx_set = range(5, 10)
		#idx_set = [3]
		cutted_ques = []
		for idx, dialouge in enumerate(test_question):
			tmp_ls = []
			for sentence in dialouge:
				#print(type())
				cutted_ls = jieba.lcut(sentence, cut_all=False)
				tmp_ls = tmp_ls + cutted_ls
			cutted_ques.append(tmp_ls)
		
	#	print(cutted_ques)
		#print(test_question[6])
		#print(jieba.lcut(str(test_question[5])))

		
		for idx in idx_set:
   			

			#tags_question = jieba.lcut(str(test_question[idx]), cut_all = args.w2v_train_cutall)
			#tags_question = jieba.posseg.cut(str(test_question[idx]))
			#print(average_sentence(tags))
			#tmp_question = average_sentence(tags_question)
			print(test_question[idx])
			max_sim = 0
			tmp_idx = 0
			for idx_ans, ans in enumerate(test_ans[idx]):
				#tags_ans = jieba.analyse.extract_tags(str(ans))
				cutted_ans = jieba.lcut(ans, cut_all=args.w2v_train_cutall)
				#print(cutted_ans)
				sim = sim_sentences(average_sentence(cutted_ans), average_sentence(cutted_ques[idx]))
				if sim > max_sim:
					tmp_idx = idx_ans
					max_sim = sim
			
			print(test_ans[idx][tmp_idx], " : ", max_sim)

			print()
				
	if args.action == 'test':

		model_paths = []
		model_dim = []
		for idx_path in range(1, 9):
			model_paths.append(args.w2v_model_load_path+str(64*idx_path))
			model_dim.append(64*idx_path)

		print(model_paths)
		if args.w2v_train_cutall:
			cutted_que_path = './cutted_ques_cutall.pkl'
			cutted_ans_path = './cutted_ans_cutall.pkl'
			pre_path = './prediction/prediction_cutall_{}.csv'
		else:
			cutted_que_path = './cutted_ques_nocutall.pkl'
			cutted_ans_path = './cutted_ans_nocutall.pkl'
			pre_path = './prediction/prediction_nocutall_{}.csv'

		for idx_path, path in enumerate(model_paths):
			model = Word2Vec.load(path)
			print("vocab size is ", len(model.wv.vocab))
			#print(model.most_similar("怎麼辦"))
			print("testing for dim = ", model_dim[idx_path])
			cutted_ques = []
			cutted_ans = []
			if idx_path == 0:
				df = pd.DataFrame(pd.read_csv(args.testing_data_path,encoding='utf-8'))
				test_id = df.values[:, 0]
				test_question = preprocess_testing_data(df.values[:, 1])
				test_ans = preprocess_testing_data(df.values[:, 2])
				
				#idx_set = range(5, 10)
				#idx_set = [3]
				
				for idx_ques, dialouge in enumerate(test_question):
					tmp_ls = []
					for sentence in dialouge:
						#print(type())
						cutted_ls = jieba.lcut(sentence, cut_all=args.w2v_train_cutall)
						tmp_ls = tmp_ls + cutted_ls
					cutted_ques.append(tmp_ls)
				pkl.dump(cutted_ques, open(cutted_que_path, 'wb'))
				
				
				for idx_ans, ans in enumerate(test_ans):
					tmp_ans = []
					for each_ans in ans:
						tmp_ans.append(jieba.lcut(each_ans, cut_all=args.w2v_train_cutall))
					cutted_ans.append(tmp_ans)
				pkl.dump(cutted_ans, open(cutted_ans_path, 'wb'))

			else:
				cutted_ques = pkl.load(open(cutted_que_path, 'rb'))
				cutted_ans = pkl.load(open(cutted_ans_path, 'rb'))
				print("data amount is", len(cutted_ques))

			#print(len(cutted_ques))
			#print(cutted_ans[-1])
		#	print(cutted_ques)
			#print(test_question[6])
			#print(jieba.lcut(str(test_question[5])))
			
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
					if sim > max_sim:
						tmp_idx = idx_ans
						max_sim = sim
				#print(tmp_idx)
				print(max_sim)
				option.append(tmp_idx)
				#print(option)
				#print(test_ans[idx][tmp_idx], " : ", max_sim)
			#print(option)
			print()
			data_id = np.linspace(1, len(option), len(option)).astype(int)
			df_output = pd.DataFrame(option, columns = ["ans"], index = data_id)
			df_output.to_csv(pre_path.format(str(64*idx_path)), index_label = 'id')
			
if __name__ == '__main__':
	main()