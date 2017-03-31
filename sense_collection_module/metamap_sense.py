#! usr/bin/env python
#! -*- coding -*- utf-8

'''main_op is the entry function for this module. 
it uses a python wrapper for metamap to extract list of concepts for a given text.
_get_metamap_resolution is the function that takes a text input and retrieve the related concepts from it.
its better to give bigrams as input to the metamap. metamap require a text context to retrieve relevant text
concepts.
Extracting concepts from metamap can be very slow. so its better to run it in parallel.
install it in several machine and run it in all machine parallel. also do the task level parallel execution.
Need to work with the HPC administrator to install software in multiple machines.
'''
import math
import sys
import os
import nltk
from collections import Counter
from collections import defaultdict
import nltk
from nltk import word_tokenize
from pymetamap import MetaMap
from collections import defaultdict
import re
global_metamap = None

def read_and_load(path_name = None, file_name = None):
	path = os.path.join(path_name,file_name)
	with open(path,'r+') as data_reader:
		data = data_reader.readlines()
	return data

def regular_expression_ops(string):
	return re.findall('[a-zA-Z0-9 ]+',string)[0]
# metamap object require to map each word to its concepts
# following function loads the MetaMap object from directory where metamap installed
# bin directory contains binary executable for metamap
def load_metaMap():
	global global_metamap
	global_metamap =  MetaMap.get_instance('../../../../../../opt/public_mm/bin/metamap12')

# following function loads the metamap object and start extracting list of concepts for given texts
# it actually retrieve related concepts and the score of relevancy, preferred naming for that concept. 
def _get_metamap_resolution(text):
	global global_metamap
        '''if(metamap_ob == None):
	   print("Object need to have a valid reference\n")
	   return'''
	metamap_ob = global_metamap#load_metaMap()
	try:
		concepts,error = metamap_ob.extract_concepts([text.strip()])
		if(len(concepts) == 0):
			return None
		else:
			return concepts
	except e:
		print e
	return

def main_ops():
	dic = defaultdict()
	adr_list = []
	# directory name and filename as a parameter to load data from disk
	fdr_data = read_and_load(path_name = 'FDR/meddra_all_se.tsv',file_name = 'meddra_all_se.tsv' )
	for each in fdr_data:
		dic[each.split('\t')[5].lower()] = each.split('\t')[2]
	adr_data = read_and_load(path_name = 'sabbir', file_name = 'neg_conjunction_data_02.txt')
	for each in adr_data:
		adr_list.append(regular_expression_ops(each.split('\t')[0]).lower())
	metamap_instance = load_metaMap()
	print(len(list(dic.iteritems())))
	with open('output/preferred_name.txt','w+') as data_writer:
		for key in adr_list:#dic.iteritems():#adr_list:
			concepts,error = metamap_instance.extract_concepts([key])
			#data_writer.write(each+'\t')
			for sub_concept in concepts:
				'''data_writer.write'''
				print(key.strip()+'\t'+ sub_concept.score.strip()+'\t'+sub_concept.preferred_name.strip()+'\t' +sub_concept.cui.strip()+'\n')


	print(str(len(fdr_data))+ ' ' + str(len(adr_data)))
	
	return
# following function to perform a matching between two documents using related concept sets.
def matching_ops():
	fdr_dic = defaultdict(list)
	fdr_data = read_and_load(path_name = 'output',file_name = 'fdr_preferred_name.txt' )
	for each in fdr_data:
		sub_each = each.split('\t')
		fdr_dic[sub_each[0].lower()].append((sub_each[1].lower(),sub_each[2].lower(),sub_each[3].lower()))
	metamap_data = read_and_load(path_name ='output' ,file_name = 'preferred_name.txt')
	dic_data = defaultdict(list)
	for each in metamap_data:
		sub_each = each.split('\t')
		dic_data[sub_each[0].lower()].append((sub_each[1].lower(),sub_each[2].lower(),sub_each[3].lower()))
	with open('output/matched03.txt','w+') as data_match_writer, open('output/mismatched03.txt','w+') as data_mismatched_writer:
		list_cui = defaultdict(list)
		for each in dic_data.items():
			settings = 0
			for sub_each in each[1]:
				for fdr in fdr_dic.iteritems():
					for sub_fdr in fdr[1]:
						if(sub_fdr[2].strip() == sub_each[2].strip()): #or fdr[1][1].strip() == sub_each[2].strip()):
							#data_match_writer.write(each[0]+'||'+ fdr[0]+'||'+sub_fdr[1]+'||'+sub_fdr[2].strip()+'\n')
							list_cui[each[0]].append((fdr[0],sub_fdr[0],sub_fdr[2]))
							settings = 1
							break
					#if(settings == 1):
					#	break
				#if(settings == 1):
					#break
			if(settings == 0):
				data_mismatched_writer.write(each[0]+'\n')
		for each in list_cui.iteritems():
			sort_ = sorted(each[1],key=lambda x:x[1],reverse=True)
			data_match_writer.write(each[0]+'||'+sort_[0][0]+ '||' + sort_[0][1] +'||'+sort_[0][2].strip()+'\n')

	return
if __name__ == '__main__':
	#pass
	main_ops()
#	matching_ops()
