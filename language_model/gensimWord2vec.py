import nltk
import gensim
import re
import math
import os
import sklearn
import numpy as np
from gensim.models import word2vec
from gensim.models import Phrases
from sklearn.feature_extraction.text import CountVectorizer
from nltk import RegexpTokenizer
from nltk import WordPunctTokenizer
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import logging
import gensim.models.doc2vec
from preprocess_word_to_vec import iterate_main_ops
from nltk.corpus import stopwords
from sklearn import decomposition
import itertools
st_words = stopwords.words('english')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
list_Of_sentences = []
model = None
class modelParam(object):
	def __init__(self,dim = 100 ,context_size = 5,word_count = 4,workers = 1,name = 'model/gensim0',phrase_model = False):
		self.dim = dim
		self.context_size = context_size
		self.word_count = word_count
		self.workers = workers
		self.name = name
		self.phrase_model = phrase_model
		return	
class mySentences(object):
	def __init__(self,dirname = None,phrase_model = False):
		self.dir_name = dirname
		self.tokenizer = word_tokenize
		self.list_of_stripper = [',','.','\'','"']
		self.stemmer = SnowballStemmer("english")
		self.phrase_model = phrase_model
		self.count_pass = 0
		return
	def __repr__(self):
		return 'directory name %s' % (self.dir_name)
		return
	def __iter__(self):
		for fnames in os.listdir(self.dir_name):
			for idx, line in enumerate(open(os.path.join(self.dir_name,fnames))):
				line = line.lower()
				elem = self.tokenizer(line)#self.tokenizer.tokenize(line)
			# if the data is already tokenized and formatted then deactivate the following 14 lines
				'''
				for each in self.list_of_stripper:
					for datum_r in xrange(0,len(elem)):
						elem[datum_r] = elem[datum_r].strip(each)
				try:
					elem = [self.stemmer.stem(token.decode('utf-8','ignore')) for token in elem if re.match('[A-z0-9]+',self.stemmer.stem(token.decode('utf-8','ignore')),re.IGNORECASE) != None and len(self.stemmer.stem(token.decode('utf-8','ignore'))) > 2 ]
				except Exception, e:
				 	logging.basicConfig(level = logging.ERROR)
				 	logger = logging.getLogger(__name__)
				 	logger.error('exception message is ', exc_info = True)
				 	print('pass executing due to exception:- ')
				 	logger.error('elem is %s ',str(elem))
				 	self.count_pass += 1
				 	logger.error('count pass: %s', str(self.count_pass))
					#print ('exception message is: ' + str(e))
				'''
				if(self.phrase_model == False):
					yield elem
				else:
				 	try:
				 		yield gensim.models.doc2vec.LabeledSentence(words = elem,labels = ['SENT_%s' % idx])
				 	except Exception, e:
				 		logging.basicConfig(level = logging.ERROR)
				 		logger = logging.getLogger(__name__)
				 		logger.error('exception message is: ', exc_info = True)
				 		
				 	
		return
def phrasesModel(parameter = None):
	sentences = ['who we are to deal in San Francisco','do not go down','put down the gun and freeze','machine learning can be fun']
	list_to_work = []
	#for ind,each in enumerate(sentences):
	#	list_to_work.append(word_tokenize(each))
	bigram_sentences = Phrases(sentences)
	print(bigram_sentences)
	for each in sentences:
		print(bigram_sentences[each])
	return
def readSentences():
	global list_Of_sentences
	file_names = 'sentence'
	for i in xrange(26):
		file_To_read = open('sentence/'+file_names+str(i),'r')
		try:
			data = file_To_read.readlines()
		finally:
			file_To_read.close()
		list_Of_sentences.extend(data)
		
	return
def createDocModelandSave(parameters,version = 0.2):
	my_sentence = mySentences('sentence',phrase_model = True)#('sentence_abs_tit/',phrase_model = True)
	model = gensim.models.doc2vec.Doc2Vec(min_count = 8, size = 200,window = 10,alpha = 0.025,min_alpha = 0.025,workers = 24)
	model.build_vocab(my_sentence)
	#for idx in xrange(2):
	model.train(my_sentence)
	#	model.alpha -= 0.002
#		model.min_alpha = model.alpha
        model.save('doc_model/model' + str(version))
	return

def createModelandSave(parameters):
	global list_Of_sentences
	global model
	print('running word vector model')
	my_sentence = mySentences(dirname = 'mysql_scripts/concept_text_dir/')

	model = word2vec.Word2Vec(my_sentence,size = parameters.dim ,window = parameters.context_size,min_count = parameters.word_count, workers = parameters.workers)
	model.save(parameters.name)
	try:
		file_name_of_model = open('fileName/'+parameters.name,'w+')
	except IOError:
		print('No directory or file name of that name')
		return
	file_name_of_model.write(parameters.name)
	file_name_of_model.close()
	print('Successfully created the model')
	
	return
def createAndsavePhrasemodel(parameter):
	globalllist_Of_sentences
	global model
	my_sentences = mySentence('sentence')
	model = word2vec.Word2Vec(my_sentence,size = parameters.dim, window = parameters.context_size,min_count = parameter.word_count ,workers = parameters.workers)
	return
def recoverModel():
	global list_Of_sentences
	global model
	data = input('write the model name to read: ')
	file_To_read = open('fileName/' + data)
	try:
		file_name = file_To_read.readline()
	finally:
		file_name.close()
	model = word2vec.Word2Vec.load('fileName/'+file_name)
	return model
def load_and_perform_molecular_sim(path_name = 'word_to_vec_1.0'):
	model = word2vec.Word2Vec.load(path_name)#('word_to_vec_1.7')
	return model
def max_match_finder(val_,side_eff_list,side_eff_names):
	max_value = -1.0
	scoring_list = []
	for ind,value in enumerate(side_eff_list):
		add_up = []
		for sub_val_ in val_:
			max_value = -1.0
			for sub_value in value:
				v_ = np.dot(sub_val_,sub_value)
				max_value = max(v_, max_value)
			add_up.append(max_value)
		if(len(add_up) != 0 ):
			scoring_list.append((sum(add_up)/float(len(add_up)),ind))
	print(len(scoring_list))
	scoring_list = sorted(scoring_list,key = lambda x:x[0],reverse = True )
	print(str(scoring_list[0]))	
	print(str(len(side_eff_names)))
	return (side_eff_names[scoring_list[0][1]].strip(),scoring_list[0][0],scoring_list[len(scoring_list)-1][0])

def get_pca(model = None):
	
	x_writer = open('../molecular_docking/output/matched_l.txt','w+')
	y_writer = open('../molecular_docking/output/matched_r.txt','w+')

	with open('../molecular_docking/output/matchingofmismatch02.txt') as data_reader,open('../molecular_docking/output/pair_reduced_dim.txt','w+') as data_writer:
		data = data_reader.readlines()
		list_data = []
		for each in data:
			datum = each.split('||')[:2]
			list_data.append(datum)
		pca_ = decomposition.PCA(n_components = 2)		
		#reduced_dim
		for each in list_data:
			x_writer.write(each[0].lower()+'\n')
			y_writer.write(each[1].lower()+'\n')
		x_writer.close()
		y_writer.close()
		side_name_l = open(os.path.join('../molecular_docking/output/','matched_l.txt'))
		side_name_r = open(os.path.join('../molecular_docking/output/','matched_r.txt'))
		objs_l = iterate_main_ops(path_name='../molecular_docking/output/',file_name='matched_l.txt')
		objs_r = iterate_main_ops(path_name='../molecular_docking/output/',file_name='matched_r.txt')
		exception_list_l = []
		exception_list_r = []
		name_l = []
		name_r = []
		index = 0
		for each in zip(objs_l,objs_r,side_name_l,side_name_r):
			temp_list_l = []
			temp_list_r = []
			count = 0
			n_str =''
			for ind_,val_ in enumerate(each[0]):
				try:
					temp_list_l.append(model[val_])
					n_str += ' ' + val_
					count += 1
				except Exception,e:
					#exception_list.append(value)
					print(e)
			if(count != 0):
				temp_vec = sum(temp_list_l)/float(len(temp_list_l))
				if(np.isnan(temp_vec).any() or np.isinf(temp_vec).any()):
					temp_vec = np.nan_to_num(temp_vec)
					temp_vec = np.ins_to_num(temp_vec)
				exception_list_l.append(temp_vec)
				name_l.append(each[2])
			else:
				continue
			count = 0
			n_str =''
			for ind,val_ in enumerate(each[1]):
				try:
					temp_list_r.append(model[val_])
					n_str += ' ' + val_
					count+=1
				except Exception,e:
					print(e)
			#print(temp_list)
			#print(exception_list_l[index])
			if(count == 0):
				del exception_list_l[len(exception_list_l) - 1]
				del name_l[len(name_l)-1]
				continue
			index+=1
			temp_vec = sum(temp_list_r)/float(len(temp_list_r))
			if(np.isnan(temp_vec).any() or np.isinf(temp_vec).any()):
				temp_vec = np.nan_to_num(temp_vec)
				temp_vec = np.inf_to_num(temp_vec)
			#if(np.isnan(temp_vec).any() or np.isinf(temp_ve).any()):
			#	temp_vec = np.nan_to_num(temp_vec)
			exception_list_r.append(temp_vec)
			name_r.append(each[3])

		pca_.fit(list(itertools.chain(exception_list_l,exception_list_r)))
#a.extend(exception_list_r))
		exception_list_l = pca_.transform(exception_list_l)
		exception_list_r = pca_.transform(exception_list_r)
		index = 0
		for each in zip(exception_list_l,exception_list_r):
			data_writer.write(name_l[index].strip()+'\t'+str(each[0][0])+' '+str(each[0][1])+'\t'+name_r[index].strip()+'\t'+ str(each[1][0])+' '+str(each[1][1])+'\n')
			index += 1


	return
def perform_sense_resolution(model = None):
	if(model == None):
		print('model is not loaded')
		return
	with open('mismatched03.txt','r+') as data_reader,open('meddra_all_se.tsv/meddra_all_se.tsvm','r+') as sideef_reader,open('../molecular_docking/output/exception_list.txt','w+') as ex_writer:
		data = data_reader.readlines()
		side_effect = sideef_reader.readlines()
		objs = iterate_main_ops(path_name='',file_name='mismatched03.txt')
		sideef_obs = iterate_main_ops(path_name = 'meddra_all_se.tsv',file_name='meddra_all_se.tsvm')
		mismatch_list = list()
		sideef_list = list()
		exception_list = []
		for ind,value in enumerate(objs):
			temp_list = []
			count_ = 0
			for ind_,val_ in enumerate(value):
				try:
					temp_list.append(model[val_])
					count_ += 1
				except Exception,e:
					#exception_list.append(value)
					print(e)
			#print(temp_list)
			if(count_ == 0):
				exception_list.append(value)
			else:
				mismatch_list.append(temp_list)
		for each in exception_list:
			ex_writer.write(' '.join(each).strip()+'\n')
		for value in sideef_obs:
			temp_list = []
			for val_ in value:
				try:
					temp_list.append(model[val_])
				except Exception,e:
					print(e)
			sideef_list.append(temp_list)
		#print(str(len(sideef_list)))
		#print(str(len(side_effect)))
		with open('../molecular_docking/output/matchingofmismatch03.txt','w+') as data_writer:
			for ind, val in enumerate(mismatch_list):
				if(len(val)!=0):
					matched, b_score,w_score = max_match_finder(val,sideef_list,side_effect)
					data_writer.write(data[ind].split('\t')[0].strip() + '||' + matched.strip() +'||'+ str(b_score) +'||' + str(w_score) + '\n')
				
		
	return
def mostSimilarmatch():
	global model
	posList = []
	
	posList(input('Enter two words for vector addition: or None to exit'))
	while(postList[0] != None):
		posList.append(input('Enter another one: '))
		negList = list(input('Enter negative word: '))
		print(model.most_similar(posList,negList)+'\n')
		posList.clear()
		posList.append('Enter two words for vector addition: or None to exit')
		
	return
def cosineSimilarityCheck():
	global model
	data1 = input('Enter first word: ')
	data2 = input('Enter second word: ')
	while(data1 != None and  data2 != None):
		print('result of cosine similarity: '+float(model.similarity(data1,data2)))
		data1 = input('Enter first word: ')
		data2 = input('Enter second word: ')

	return
def doesNotMatch():
	global model
	sentence = input('Enter a sentence: ')
	while(sentence != None):
		print('Result of mismatch: '+ float(sentence))
		sentence = input('Enter a sentence')
	return
# vector model 1.5 is using old data formatting, tokenization and stripping
# vecgtor model 1.7 is using new stop word list and formatting
def  __main__():
	doc_models = 0
	dim = 300#input('Enter dimension of word vectors: ')
	word_context = 10#input('Enter word Context size should be between 5-10: ')
	min_count = 10#input('Enter word of minimum freq to ignore: ')
	workers = 30#input('number of threads to use: ')
	name = 'concept_to_vec_1.1'#input('Enter the name of the module: ')
 #	print(str(dim) + str(word_context)+ str(min_count)+str(workers))
	parameters = modelParam(dim,word_context,min_count,workers,name)
	print('dimension '+str(parameters.dim)+' word context '+ str(parameters.word_count))
	#readSentences()
	if(doc_models == 0):
		print('doc model is about to run')
		createModelandSave(parameters)
	else:
		print('word model is about to run')
		createDocModelandSave(parameters)
	return
if __name__ == '__main__':
	#perform_sense_resolution(load_and_perform_molecular_sim())
	#get_pca(load_and_perform_molecular_sim())
	__main__()
