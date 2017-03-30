#! ~/anaconda2/bin/python
#! -*- coding:utf-8 -*-
import nltk
import gensim
import re
import math
import sklearn
import os
import codecs
import itertools
import tempfile
import joblib
from joblib import load,dump
from gensim.models import word2vec
import numpy as np
#import gensim.models import Phrases
from nltk import RegexpTokenizer
from nltk import  WordPunctTokenizer
from joblib import Parallel,delayed
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
import logging
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import gensim.models.doc2vec
from nltk.corpus import stopwords
import sys
sys.path.insert(0,'../')
from gensimWord2vec import load_and_perform_molecular_sim
#from metamap_sense import load_metaMap
#from metamap_sense import  _get_metamap_resolution
from nltk.util import ngrams
from collections import defaultdict
from collections import Counter
from collections import deque
from random_walk import build_graph
from network_flow import mainOps
from optparse import OptionParser
from multiprocessing import Pool
from multiprocessing import Lock, Manager
st_words = stopwords.words('english')
#sys.path.insert(0,'../../molecular_docking/')
class sentenceParser(object):
   def __init__(self,dir_name = None,text_id = None,splitting_ch = '\t',file_ind = None,special_fn = None,start=None,end=None):
      self.dir_name = dir_name
      self.key_word = None
      self.tokenizer = word_tokenize
      self.list_of_stripper= [',','.','\'','"']
      self.stemmer = PorterStemmer()
      self.count_pass = 0
      self.text_ind = text_id
      self.splitting_ch = splitting_ch
      self.special_fn = special_fn
      self.file_ind = file_ind
      self.start = start
      self.end = end
      return
   def _get_formatted_data(self,data):
      data = self.tokenizer(data)
      return data
   def __iter__(self):
      stop_words = []
      for fnames in os.listdir('stop_words/'):
	 if(fnames.find('special') != -1):
	    special_ch = open(os.path.join('stop_words',fnames)).readlines()
	 else:
	    stop_words += open(os.path.join('stop_words',fnames)).readlines()
      special_ch = [each.strip() for each in special_ch]
      stop_words = [each.strip() for each in stop_words]
      count_here = 0
      for fnames in self.dir_name:#os.listdir(self.dir_name):
	#if(self.file_ind != None):
		#if(fnames.find(str(self.file_ind)) == -1):
			#continue
	 if(count_here == 1 ):
		break
	 else:
	  	count_here += 1
	 if(self.special_fn != None):
	    if(fnames.find(self.special_fn) != -1):
	       continue
	 count__ = 0
	 for idx, line  in enumerate(self.dir_name):#enumerate(open(os.path.join('fda_dataset_02/',fnames))):
            if(idx < self.start or self.end < idx):
		    continue
	    count__ += 1
   	    line = line.lower()
	    splitted_line = line.split(self.splitting_ch)
	    self.key_word = splitted_line[0]
	    if(isinstance(self.text_ind,list)):
	       list_of_items = []		
	       for datum in self.text_ind:
	          list_of_items.append(self._get_formatted_data(splitted_line[datum]))
	       yield list_of_items
	    else:
   	       subsplitted_line = splitted_line[self.text_ind]
	       side_effect_name = splitted_line[0]
	       print(side_effect_name)
	       list_of_subsplitted_lines = self.tokenizer(subsplitted_line)
	       elem = [tokens for tokens in list_of_subsplitted_lines if tokens not in stop_words]
	       for datum_ in special_ch:
		  elem = [tokens for tokens in elem if tokens not in re.findall(datum_,tokens)]
	       elem = [tokens for tokens in elem if len(tokens) > 2]
	       yield (elem,side_effect_name,count__)#,side_effect_name)
		   
def accumulate(dir_name= '../../molecular_docking/dataset/mrrel/',file_name = 'mrrel.txt'):
	with codecs.open(os.path.join(dir_name,file_name)) as data_reader:
		data = data_reader.readlines()
		#sub_data = [(datum[1],datum[2]) for datum in data]
		cnt = defaultdict(dict)
		total_count = defaultdict(int)
	        actions = 0
      	 	try:
      			cnt = pickle.load(open(os.path.join('pickled_dir/','cnt'),'rb'))
			total_count = pickle.load(open(os.path.join('pickled_dir/', 'total_cnt'),'rb'))
      		except:
			actions = 1
     		if actions == 1:

			for datums in data:
				datum = datums.split('\t')
				if(cnt[datum[0]].get(datum[4]) == None):
					cnt[datum[0]][datum[4]] = 1
					cnt[datum[4]][datum[0]] = 1
				else:
			 		cnt[datum[0]][datum[4]] += 1
			 		cnt[datum[4]][datum[0]] += 1
				total_count[datum[0]] += 1
				total_count[datum[4]] += 1
			pickle.dump(cnt,open(os.path.join('pickled_dir/','cnt'),'wb'))
	        	pickle.dump(total_count,open(os.path.join('pickled_dir/','total_cnt'),'wb'))

	return (cnt,total_count)
def get_score(graph = None):

	if(graph == None):
	   print('graph is null and exiting')
	   return
	node_result = defaultdict(int)
	for node in graph._graph_dict:
		for each in graph.indegree_list[node]:
	   		graph._graph_dict[node][1].score += graph._graph_dict[each][0][node]
	      
        print('total nodes with nonzero indegree: ' + str(len(list(node_result.iteritems()))))	
	return node_result
def func(p):
	return p[1]
def assign_max_score(node_result = None,node_tracker = None):
   result_dict = defaultdict(list)
   max_dict = {}
   for each in node_result._graph_dict:
      if(node_tracker.get(each.strip()) == None):
	      print('I am none for ' + str(each))
	      continue
      keys = node_tracker[each.strip()]
      res = node_result._graph_dict[each][1].PR
      w_res = node_result._graph_dict[each][1].weighted_PR
      s = node_result._graph_dict[each][1].score
      result_dict[keys].append((s,res,w_res,each.strip()))
   for each in result_dict:
      max_dict[each] = max(result_dict[each],key = func)
   return max_dict
def build_concept_network(mrrel_file = '../../molecular_docking/dataset/mrrel/mrrel.txt'):
	matrix = defaultdict(dict)
	with codecs.open(mrrel_file) as data_reader:
		data = data_reader.readlines()
		for each in data:
			datum = each.split('\t')[0:5]
			matrix[datum[0]][datum[4]] = 1
			matrix[datum[4]][datum[0]] = 1
	return matrix
def load(data_path = None,des_path = None,ind_ = None):
   if(data_path == None):
      print("data set is empty returning ....")
      return
   #metamap_ob = load_metaMap()
   start = (ind_- 1)*11
   end = start + 11
   ''' 
   print('i am process number %d' % ind_)
   print('my start index is %d' % start)
   print('my end index is %d' % end)
   '''
   (concept_net,total_cnt) = (None,None)# accumulate()
   #print('After loading concepts\n')
   #g_concept_net = Graph(graph_dict= concept_net)
   parserObject = sentenceParser(dir_name = data_path,text_id = 3, splitting_ch = '\t',file_ind = ind_,start = start,end = end)
   for ind,each in enumerate(parserObject):
      #if(s_name.find(str(ind_)) == -1):
      #	      continue
      actions = 0
      try:
	#bigrams = pickle.load(open(os.path.join('pickled_dir/',each),'rb'))
	bigram_dic = pickle.load(open(os.path.join('pickled_dir_04/', each[1]+'_dic'),'rb'))
	#print(len(bigram_dic))
	bigram_list = pickle.load(open(os.path.join('pickled_dir_04/', each[1]+'_list'),'rb'))
	#for each in bigram_list:
	#	if(parseObject.key_word.find(each) != -1):
	#		print(each)
	#print('after loading bigram dictionary')
	#print('I am executing right now and no exception\n')
      except:
        print('Exception detected in %d' % ind)
	print('exception for %s' % parserObject.key_word)
	actions = 1
      li = parserObject.key_word
      #li = '_'.join(li)
      if(li.find('/') != -1):
	li = '_'.join(li.split('/'))

      if actions == 1:
      	#sentence = ' '.join(each[0])
 	#for some reason there is something wrong with this lines
        print('%d' % ind_)
	print('%s' % each[1])
	print('%d' % each[2])
	bigrams = ngrams(each[0],2)
	bigram_dic = defaultdict(list)
	bigram_list = []
	load_metaMap()
	for datum in bigrams:
	   concept_list = _get_metamap_resolution(' '.join(datum))
	      #print(concept_list)
	      #print('\n')
	   if(concept_list == None):
	      continue
	      #bigram_list.append(' '.join(datum))
	   for subconcept in concept_list:
		bigram_dic[' '.join(datum)].append(subconcept.cui.strip())
	      	bigram_list.append(' '.join(datum))
	   #pickle.dump(bigrams,open(os.path.join('pickled_dir/',each),'wb'))
	pickle.dump(bigram_dic,open(os.path.join('pickled_dir_04/',str(li + '_dic')),'wb'))
	pickle.dump(bigram_list,open(os.path.join('pickled_dir_04/',str(li + '_list')),'wb'))
	# remove this continuation 
	# temporary just to generate the dictionary and list
      with codecs.open(os.path.join('fda_output_score_04/',li),'w+') as data_writer:
      	(g,node_tracker) = build_graph(g = bigram_list,label_list = bigram_dic,concept_net = concept_net,total_cnt = total_cnt,cosine = 1)
      	#print(node_tracker)
      	data_writer.write('Key word' + '\t'+ str(parserObject.key_word) + '\n' )
      	data_writer.write('Total Edge:' + '\t' + str(g.total_edge)+ '\n')
      	data_writer.write('Total Nodes:'+ '\t' + str(len(list(node_tracker.iteritems()))) + '\n')
	   #node_result = get_score(graph = g)
      	g._generate_indegree_outdegree()
      	g.init_rank() 
      	g.iteration = 10
      	counter = 0
     	while(True):
      		if counter == g.iteration:
			break
		counter += 1
		g.calculate_page_rank()
		g.calculate_weighted_page_rank()
      	node_result = get_score(graph = g)
        max_dict = assign_max_score(node_result = g,node_tracker = node_tracker)
        for key,values in max_dict.iteritems():
      		data_writer.write(str(key) + '\t' + str(values[0])+'\t' +str(values[1]) + '\t' +str(values[2]) + '\t' + str(values[3]) + '\n')
      
   return
def handle_maxflow_ops(node_list_a,list_of_node_list_b,edge_weight,model,name,start,end,locker):
	result  = []
	for each in list_of_node_list_b:
		max_flow = mainOps(node_list_a = node_list_a,node_list_b = list_of_node_list_b[each],edge_weight = edge_weight,pair_dist_mat = model)
		result.append((each,max_flow))
	locker.acquire()
	if not os.path.exists(os.path.join('output_dir/',name.strip())):
		os.makedirs(os.path.join('output_dir/',name.strip()))
	#with open(os.path.join('output_dir/',name.strip() + str(index)),'w+') as data_writer:
	for each in result:
		with codecs.open(os.path.join(os.path.join('output_dir/',name.strip()),name.strip()+str(start)+'_'+str(end)),'a+') as data_writer:
			data_writer.write(str(each[0].strip() )+ '\t' + str(each[1]) + '\n')
		
	locker.release()
	return result
def utilize_network_flow(source_path = None, des_path = None,rank_index = 1,start = 1,end = 1):
	element_list = defaultdict(list)
	target_element_list = defaultdict(list)
	#print(type(source_path))
	end = len(source_path) if int(end) > len(source_path) else int(end)
	sub_source_path = source_path[int(start)-1:int(end)]
	for each in sub_source_path:
		with open(os.path.join('fda_output_score_weight_02/',each)) as data_reader:
			data =data_reader.readlines()
			key_word = data[0].split('\t')[1].strip()
			max_len = 30 if len(data) >= 33 else len(data)
			for datum in data[3:max_len]:
				cui = datum.split('\t')[4]
				r_val = datum.split('\t')[rank_index]
				element_list[key_word].append((cui,float(r_val)))
			for each in element_list:
				element_list[each] = sorted(element_list[each], key = lambda x:x[1],reverse = True)
	for each in os.listdir(des_path):
		with open(os.path.join(des_path,each)) as data_reader:
			data = data_reader.readlines()
			key_word = data[0].split('\t')[1].strip()
			max_len = 30 if len(data) >= 33 else len(data)
			for datum in data[3:max_len]:
				cui = datum.split('\t')[4].strip()
				r_val = datum.split('\t')[rank_index]
				target_element_list[key_word].append((cui,float(r_val)))
			target_element_list[key_word] = sorted(target_element_list[key_word],key = lambda x:x[1],reverse = True)
	contain_elements = 30 if len(element_list.values()) > 30 else len(element_list.values())
	t_contain_elements = 30 if len(target_element_list.values()) > 30 else len(target_element_list.values())
	source_list = list(itertools.chain.from_iterable([each[:contain_elements] for each in element_list.values()]))
	target_list = list(itertools.chain.from_iterable([each[:t_contain_elements] for each in target_element_list.values()]))
	final_list = target_list + source_list			
	final_list = np.array(final_list)
	index_dict = {}
	model = load_and_perform_molecular_sim(path_name = 'concept_to_vec_1.1')
	if(model == None):
		print('there is something wrong with model loading')
		return
	pair_dist_list = []

	
	#for ind,each in enumerate(final_list):
	#	index_dict[each[0]] = ind
	#	pair_dist_list.append(model[each[0].strip().lower()])
	#pair_dist_list = np.array(pair_dist_list)
		
	#action = 0
	#try:
	#	pair_dist = pickle.load(open(os.path.join('distance_matrix','distance_mat'),'rb'))#need to specify the directory to write the file
	#except:
	 #	action = 1
	#	logging.basicConfig(file_name = 'log_info.txt',filemode='w+',level=logging.INFO)
	#	logger = logging.getLogger(__name__)
	#	logging.info('setting action and creating pair distance matrix  %s', str(each))

	#print('distance matrix lenght %s ' % str(len(pair_dist_list)))

	#if action == 1 :
	#	try:
	#		pair_dist = pairwise_distances(pair_dist_list,metric= 'cosine')
	#		pair_dist.astype('float16')
	#		pickle.dump(pair_dist,open(os.path.join('distance_matrix/','distance_mat'),'wb'))#need to write down the proper directory name here
	#	except:
	#		logging.basicConfig(file_name = 'log_info.txt',filemode='w+',level=logging.INFO)
	#		logger = logging.getLogger(__name__)
	#		logging.info('get some memory error for %s', str(each))


	result_dict = defaultdict(list)
	elements = iter(element_list.iteritems())
	number = int(end) - int(start) + 1
	list_elements = []
	while(True):
		temp_dict = dict(itertools.islice(elements,number))
		if len(temp_dict) == 0:
			break
		list_elements.append(temp_dict)
	manager = Manager()
	lock = manager.Lock()
	temp_dir = tempfile.mkdtemp()
	filename = os.path.join(temp_dir,'temp.mmap')
	if not os.path.exists(filename):
		#os.unlink(filename)
		_=joblib.dump(lock,filename)
	locker = joblib.load(filename,mmap_mode='r+')
	for each in target_element_list.items():
		Parallel(n_jobs = 200)(delayed(handle_maxflow_ops)(each[1],elem,0.1,model,each[0].strip(),start,end,locker) for (elem,index)  in zip(list_elements,xrange(1,5000)) )
		#result_dict[each] = list(itertools.chain.from_iterable(get_result))
		#max_flow = mainOps(node_list_a = each,node_list_b = sub_each,edge_weight = 0.1,pair_dist)
		#result_dict[each] = sorted(result_dict[each],key= lambda x:x[1], reverse = True)	

	
	#for each in target_element_list:
	#	with open(os.path.join('output_dir',each),'w+') as data_writer:
	#		for element in result_dict[each]:
	#			data_writer.write(element[0] + '\t' + element[1])

	


	return
def compare_scores_and_sort(path_name = 'fda_output_score/', target = 'temp_output_dataset/activation of dopamine'):
	dir_list = os.listdir(path_name)
	fp = codecs.open(target)
	data = fp.readlines()[3:]
	model = load_and_perform_molecular_sim(path_name='../concept_to_vec_1.1')
	result_list = []
	
	matrix = defaultdict(dict)
	for each in dir_list:
		with codecs.open(os.path.join(path_name,each)) as datareader:
			data_s = datareader.readlines()
			source_data = data_s[3:]
			min_len = min(len(source_data),len(data))
			score = 0
			meta_info = data_s[:3]

			for each in data:#[:min_len]:
				if(len(each.split('\t')) < 3):
					continue
				cui_target = each.split('\t')[2].lower()
				if(model[cui_target.strip()] == None):
					continue
				max_val = -1
				track_dic = {}
				for sub_each in source_data:#[:min_len]:
				 	if(len(sub_each.split('\t')) < 3 or track_dic.get(sub_each.split('\t')[0]) != None):
						continue
					track_dic[sub_each.split('\t')[0]] = 1
					
					cui_source = sub_each.split('\t')[2].lower()
					if(model[cui_source.strip()] == None):
						continue
					if(matrix[cui_target].get(cui_source) == None and matrix[cui_source].get(cui_target) == None):
						cui_target=cui_target.strip()
						cui_source = cui_source.strip()
						max_val = max(model.similarity(cui_target,cui_source),max_val)
						matrix[cui_target][cui_source] = model.similarity(cui_target,cui_source)
						matrix[cui_source][cui_target] =  model.similarity(cui_target,cui_source)

					else:
						if(matrix[cui_target].get(cui_source) == None):
							max_val = max(matrix[cui_source][cui_target],max_val)
						else:
							max_val = max(matrix[cui_target][cui_source],max_val)

						

			        score += max_val
			result_list.append((meta_info[0].strip(),score))
 	result_list = sorted(result_list,key = lambda x:x[1],reverse = True)
	for each__ in result_list:
		print( str(each__) + '\n')
	return
def readloadreplace(pathname = '../../molecular_docking/dataset/pubmed',outputpath = 'fda_data/'):
	listdirs = os.listdir(pathname)
	for each in listdirs:
		if(each.find('fdr') == -1):
			continue
		temp_dict = {}
		with codecs.open(os.path.join(pathname,each.strip())) as data_reader, codecs.open(os.path.join(outputpath, each.strip()),'w+') as data_writer:
			data = data_reader.readlines()
			#print(len(data))
			for datum in data:
				datum_ = datum.split('\t')
				#print(len(datum_))
				if (temp_dict.get(datum_[0].strip()) == None):
					temp_dict[datum_[0].strip()] = datum_[1:]
			for key,values in temp_dict.iteritems():
			   data_writer.write(key+'\t'+values[0]+'\t'+values[1]+'\t'+values[2]+ '\n')
		


	return
if __name__ == '__main__':
	dir_list_need = os.listdir('pickled_dir_04/')
	dir_list_2 = os.listdir('temp_dataset')
	datalist = []
	start,end = sys.argv[1:]	
	track = {}
	'''
	for each in dir_list_need:
		
		#with open(os.path.join('fda_dataset_02/',each)) as data_reader:
		#	datalist += data_reader.readlines()
		
		output = re.findall('[a-zA-Z_ ]+',each)
		if(output[0].find('dic') != -1):
			string = output[0][:output[0].find('dic') - 1]
			if(string.find('/') != -1):
				string = '_'.join(string.split('/'))
			track[string] = 1
			#print(string[:output[0].find('dic')-1])
	'''
	for each in dir_list_2:
		with open(os.path.join('temp_dataset/',each)) as data_reader:
			datalist += data_reader.readlines()
	
	final_list = []
	counter = 0
	
	for each in datalist:
	  	each = each.split('\t')[0]
		if(each.find('/') != -1):
			each = '_'.join(each.split('/'))
		if(track.get(each)== None):
			final_list.append(each)
			counter += 1
			track[each] = 1
	print(counter)
	all_dir_list = os.listdir('fda_output_score_04')
	try:
		pickle.dump(all_dir_list,open('pickled_list_source_weight','wb'))
		#all_dir_list = pickle.load(open('pickled_list_source','rb'))
	except:
		#logger = logging.basicConfig(filename = 'simple_example.txt',filemode='w+')
		logger = logging.getLogger('simple_example')
		logger.setLevel(logging.DEBUG)
		# create console handler and set level to debug
		ch = logging.FileHandler(filename = 'simple_data.txt',mode = 'w+')
		ch.setLevel(logging.DEBUG)
		# create formatter
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		# add formatter to ch
		ch.setFormatter(formatter)
		# add ch to logger
		logger.addHandler(ch)
		# 'application' code
		logger.debug('latest debug message')
		logger.debug('there is an divide by zero error %s'%  sys.exc_info()[0])
		print 'log file should be created'
	utilize_network_flow(source_path = all_dir_list, des_path = 'fda_output_test/',rank_index = 2,start=start,end=end)
	#Parallel(n_jobs = 30)(delayed(load)(data_path = datalist,ind_= elem)for elem  in xrange(1,31) )

	#print(len(track))
	#print(len(final_list))
	'''
	dic = defaultdict(int)
	counter = 0
	for each in datalist:
		if(dic[each] == 0):
	
def compare_scores_and_sort(path_name = 'fda_output_score/',target = 'temp_output_dataset/activation of dopamine'):
	dir_list = os.listdir(path_name)
	fp = codecs.open(target)
	data = fp.readlines()[3:]
	model = load_and_perform_molecular_sim(path_name='../concept_to_vec_1.1')
	result_list = []
	
	matrix = defaultdict(dict)
	for each in dir_list:
		with codecs.open(os.path.join(path_name,each)) as datareader:
			data_s = datareader.readlines()
	'''
