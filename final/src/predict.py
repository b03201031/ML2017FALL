from gensim.models import Word2Vec
import pickle as pkl
import numpy as np
import time
import pandas as pd
import sys

def average_sentence(cutted_sentence, model, model_dim, freq_mean, freq_var):
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

def main():
	model_paths = []
	model_dim = []

	model_paths.append('model.h5')
	model_dim.append(256)

	print(model_paths)

	cutted_ques = []
	cutted_ans = []


	#identify testing data
	cutted_que_path = './cutted_ques.pkl' 
	cutted_ans_path = './cutted_ans.pkl'
	pre_path = sys.argv[1]

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


		freq = []
		for word in model.wv.vocab:
			freq.append(model.wv.vocab[word].count)
		freq = np.asarray(freq)
		freq_mean = np.mean(freq)
		freq_var = np.var(freq)


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
				sim = sim_sentences(average_sentence(ans, model, model_dim[idx_path], freq_mean, freq_var), average_sentence(cutted_ques[idx_ques], model, model_dim[idx_path], freq_mean, freq_var))
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
		df_output.to_csv(pre_path, index_label = 'id')


if __name__ == '__main__':
	main()