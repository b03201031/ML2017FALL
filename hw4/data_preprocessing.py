from keras.preprocessing.text import text_to_word_sequence, Tokenizer
import pickle, logging
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences


GENSIM_SIZE = 125

def file_to_list(file_path):
	with open(file_path, 'r') as f:
		label = []
		sentences = []

		for line in f:
			lineSplitted = line.split("+++$+++")
			#print(len(lineSplitted))
			if len(lineSplitted) == 1:
				sentences.append(text_to_word_sequence(line))
			else:
				label.append(lineSplitted[0])
				sentences.append(text_to_word_sequence(lineSplitted[1], filters='\n'))

		return label, sentences


def training_Gensim_model(sentences, model_path, load = False):
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	print("strat training...")
	if load:
		model = Word2Vec.load(model_path)
		
		#total_examples=model.corpus_count
		model.train(sentences,total_examples=model.corpus_count, epochs=model.iter)
		model.save(model_path)
		
	else:
		model = Word2Vec(sentences, size = GENSIM_SIZE, min_count=1)
		model.save(model_path)

	return model


def word_to_vec(sentences, model_path):

	model = Word2Vec.load(model_path)

	
	sentences_vec = []
	for sentence in sentences:

		sentence_vec = []
		count = 0
		for word in sentence:
			if count > 37:
				break																																																																																																																																			
			else:
				if word in model.wv.vocab:
					#print(type(model.wv[word]))
					sentence_vec.append(model.wv[word])
					count = count + 1
				else:																																						
					sentence_vec.append(np.zeros(GENSIM_SIZE))
					count = count + 1
		'''
		while count < 37:
			sentence_vec.append(np.zeros(200))
	
			count = count + 1
		'''
		sentences_vec.append(np.asarray(sentence_vec))		
	#print("done")																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																											
	return np.asarray(sentences_vec)

def tokenize(self, vocab_size):
	print ('create new tokenizer')
	self.tokenizer = Tokenizer(num_words=vocab_size)
	for key in self.data:
		print ('tokenizing %s'%key)
		texts = self.data[key][0]
		self.tokenizer.fit_on_texts(texts)

def BOW(data_path):
	tokenizer = Tokenizer()
	tokenizer.data
def main():

	
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	data_path = "./datas/training_label.txt"
	#data_path = "training_nolabel.txt"
	model_path = "./gensim_125_nofilter/gensim_model_125"
	

	label, sentences = file_to_list(data_path)

	training_Gensim_model(sentences, model_path)


	print("transfering word to vector")
	sentences = word_to_vec(sentences, model_path)
	label = np.asarray(label)
	label = label.astype(int)
	np.save("./preprocessed_data/label_train_125_nofilter", label)
	

	print("start to padding")
	sentences_vec = pad_sequences(sentences, maxlen=40, padding='post', dtype='float32')																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																												
	np.save("./preprocessed_data/texts_train_125_nofilter", sentences_vec)
	


	'''
	#data_path = "test.txt"
	#model_path = "test_model"
	(label, sentences_vec) = word_to_vec(data_path, model_path)
	#print("strat_to_train_1")
	#

	#(label, sentences_vec) = file_to_list(data_path_2)
	print("strat_to_save_label")
	#training_Gensim_model(sentences_vec, model_path, load=True)

	#print(label)
	#print(type(sentences_vec[1]))


	#print(sentences_vec[0].shape)

	#print(sentences_vec[0].shape)
	'''
	

	#print(sentences_vec.sentence_vechape)
#main()
