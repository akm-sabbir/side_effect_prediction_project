# usr/bin/env python
#! -*- coding -*- utf-8

import nltk
import gensim
import re
import math
from gensim.models import word2vec
import os
import codecs
from nltk import RegexpTokenizer
from nltk import WordPunctTokenizer
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
def split_and_return(line,component,splitting_char = ' '):
	return line.split(splitting_char)[component]

def preprocess_function(path_name = None,des_path = None,splitting = False,component = 1,splitting_char = ' '):
	if(path_name == None):
		raise ValueError
		return
	tokenizer = word_tokenize
	special_characters = []
	stop_words = []
	stripper = [',','.','\'','"']
	stemmer = PorterStemmer()#SnowballStemmer("english")
	for fnames in os.listdir('stop_words/'):
		if(fnames.find('special')!= -1):
			special_characters = open(os.path.join('stop_words',fnames)).readlines()
		else:
			stop_words.extend(open(os.path.join('stop_words',fnames)).readlines())
	special_characters = [each.strip() for each in special_characters ]
	stop_words = [each.strip() for each in stop_words]
	#print(str(special_characters ) + '\n')
	#print(str(stop_words) + '\n')
	for fnames in os.listdir(path_name):
		with codecs.open(os.path.join(des_path,fnames),'w+') as data_writer: 
			for idx,line in enumerate(open(os.path.join(path_name,fnames))):
				line = line.lower()
				if(splitting == True):
					line = split_and_return(line,component,splitting_char)
				elem = tokenizer(line)
				for each in stripper:
					for idx,datum in enumerate(elem):
						elem[idx] = elem[idx].strip(each)
				elem = [tokens for tokens in elem if tokens not in stop_words ]
				#print(elem)
				#print(special_characters)
				#break
				next_elem = []
				for each in elem:
				  	setting = 0
					for datum in special_characters:
				 		if(len(re.findall(datum,each)) != 0):
							setting = 1
							break
					if(setting == 0):
						next_elem.append(each)
				#print(next_elem)
				#elem = [stemmer.stem(token.decode('utf-8','ignore')) for token in next_elem if len(token.decode('utf-8','ignore')) > 2]
				sentence = ' '.join(next_elem).decode('utf-8','ignore').strip()
				#print(sentence)
				if(len(sentence) != 0):
					try:
						data_writer.write(sentence.encode('utf-8','ignore').strip() + '\n')
					except:
						 print('Ignore unicode Error')
						 pass
					

	return
class preprocessFile(object):
	def __init__(self,path_name = None,file_name = None,splitting = False,component = 1,splitting_char = ' '):
		self.path_name = path_name
		self.file_name = file_name
		self.splitting = splitting
		self.component = component
		self.splitting_char = splitting_char
		return
	def __iter__(self):
		if(self.path_name == None):
			raise ValueError
			return
		tokenizer = word_tokenize
		special_characters = []
		stop_words = []
		stripper = [',','.','\'','"']
		stemmer = SnowballStemmer("english")
		for fnames in os.listdir('stop_words/'):
			if(fnames.find('special')!= -1):
				special_characters = open(os.path.join('stop_words',fnames)).readlines()
			else:
				stop_words.extend(open(os.path.join('stop_words',fnames)).readlines())
		special_characters = [each.strip() for each in special_characters ]
		stop_words = [each.strip() for each in stop_words]
		#print(str(special_characters ) + '\n')
		#print(str(stop_words) + '\n')
		for idx,line in enumerate(open(os.path.join(self.path_name,self.file_name))):
			line = line.lower()
			line = line.strip()
			if(self.splitting == True):
				line = split_and_return(line,self.component,self.splitting_char)
			elem = tokenizer(line)
			#print(elem)
			for each in stripper:
				for idx,datum in enumerate(elem):
					elem[idx] = elem[idx].strip(each)
			elem = [tokens for tokens in elem if tokens not in stop_words ]
			#print(elem)
			#print(special_characters)
			#break
			next_elem = []
			for each in elem:
				'''setting = 0
				for datum in special_characters:
					if(len(re.findall(datum,each)) != 0):
						setting = 1
						break
				if(setting == 0):
					next_elem.append(each)'''
				next_elem.append(each)
				#print(next_elem)
			elem = [stemmer.stem(token.decode('utf-8','ignore')) for token in next_elem if len(token.decode('utf-8','ignore')) > 2]
			#print(elem)
			yield elem
				#sentence = ' '.join(elem).encode('utf-8').strip()
			#print(sentence)
			'''
				if(len(sentence) != 0):
					try:
						data_writer.write(sentence.encode('utf-8','ignore').strip() + '\n')
					except:
						print('Ignore unicode Error')
						pass
					
			'''
		return

def iterate_main_ops(path_name = None,file_name = None):
	#path_name = ''
	#file_name = 'mismathched_molecular.txt'
	ob = preprocessFile(path_name=path_name,file_name=file_name,component = 0)
	#for each in ob:
	#	pass#print(each)
	return ob
def main_ops():
	preprocess_function(path_name = 'sentence_abs_tit/',des_path = 'output_sally_sentence_abs_tit/')
	return
#if(__name__ == '__main__'):
	#main_ops()
	#iterate_main_ops()
