import sys
import codecs
from Bio import Entrez
import json
import logging
import urllib2
import mpi4py
from mpi4py import MPI
import joblib
from collections import defaultdict
from joblib import load,dump
from joblib import Parallel,delayed

def search_query(query,logging = None):
	Entrez.mail = 'asa256@uky.edu'
	results = []
	try:
		handle = Entrez.esearch(db='pubmed',sort='relevance',
		retmax='20',retmode='xml',
		term = query)
		results = Entrez.read(handle)
		if results == []:
			print None
		handle.close()
	except:
		logging.info(str(query) + 'got empty response')
	return None if len(results) == 0 else results['IdList']

def fetch_details(id_list,logging = None):
	ids = ','.join(id_list)
	Entrez.email = 'asa256@uky.edu'
	results = []
	print ids
	try:
		handle = Entrez.efetch(db = 'pubmed',retmode = 'xml',id = ids)
		results = Entrez.read(handle)
		handle.close()
	except urllib2.HTTPError:
		logging.info(str(urllib2.HTTPError) + ' , ' + str(ids))
	return results

def main_data_collector(search_text = None, start = None,end = None):
	
	logging.basicConfig(level = logging.INFO,
	format='%(asctime)s %(levelname)-10s %(message)s', filename='debug_dir/data_collector', filemode = 'w+') # this is for log files
	results = []
	try:
		for each in search_text:#[start:end]:
			if len(each) != 0 :
				results.append(search_query(each[0].strip().lower(),logging))
	except urllib2.HTTPError:
		logging.info(str(urllib2.HTTPError) +', ' + each[0].lower())
	data_dict = {}
	for idlist,name in zip(results,search_text):
		if idlist != None:
			data = fetch_details(idlist,logging)
			init_empt = defaultdict(list)
			for paper in data:
				init_empt['Article'].append( paper['MedlineCitation']['Article']['ArticleTitle'])
				try:
					for each in paper['MedlineCitation']['Article']['Abstract']['AbstractText']:
						init_empt['Abstract'].append(each)
				except KeyError:
					logging.info(KeyError)
				init_empt['Attribute'].append( name[1:])
				data_dict[name[0]] = init_empt
		else:
			data_dict[name[0]] = None 
		#print results[each]['MedlineCitation']['Article']['ArticleTitle']
		#print results[each]['MedlineCitation']['Article']['Abstract']['AbstractText']
		#print json.dumps(results[each],indent=2,separators = (',',':'))
	with open('json_output/missing_data.txt' + str(start) + str(end),'w+') as data_writer:
		json.dump(data_dict,data_writer)
	return

def calculate(abstract_data = None,article = None ):
	len_list = [len(each) for each in abstract_data]
	return sum(len_list),len(len_list),len(article)
#with codecs.open(file_name,
def json_load_and_read():
	file_list  = os.listdir('json_output/')
	keys = [each.split('\t')[0] for each in open('fdr_preferred_name_unique','r').readlines()]
	data = []
	writer = open('stat/side_effect_names.txt','w+') 
	for each in file_list:
		with open(os.path.join('json_output/',each)) as data_file:
			data = json.load(data_file)
		for key in keys:
			if data.get(key) != None:
				a,b,c = calculate( data[key]['Abstract'],data[key]['Article'])
				writer.write(key + '\t' + str(a) + '\t' + str(b) + '\t' + str(c)+ '\n')
				
	return
def main_ops(args = None):
	print 'len of argv: ' + str(sys.argv)
	arguments = sys.argv
	start,end,stride = arguments[1:]
	start = int(start)
	end = int(end) # type converted
	comm = MPI.COMM_WORLD
	size = comm.Get_size() # get size of the underlying machine
	print 'size: ' + str(size) 
	ranks = comm.Get_rank() # get rank of the underlying processor
	print 'starting'
	#logging.basicConfig(level = logging.INFO,
	#format='%(asctime)s %(levelname)-10s %(message)s',
	#filename='debug_dir/data_collector',
	#filemode = 'w+') # this is for log files
	start = ranks*stride
	end = (ranks+1)*stride
	'''
	if ranks == 0:
		end = start + int(end - start)/4
		print str(start) + ' ' + str(end)
	elif ranks == 1:
		start = start + (int(end-start)/4) + 1
		end = start  + int(end-start)/4
		print str(start) +' '+ str(end)
	elif ranks == 2:
		start = start + int(end-start)/2 + 1
		end = start + int(end-start)/4
	else:
		start = start + int(end-start)*(3/float(4)) + 1
	#return # temporary checking
	'''
	start = int(start)
	end = int(end)
	print 'rank of this process is: ' + str(ranks)
	with codecs.open('~/fdr_preferred_name_unique') as data_reader:
		search_text = [ each.split('\t') for each in  data_reader.readlines()]
	results = []
	search_text = search_text[int(start):int(end)+1 if int(end) < len(search_text) else len(search_text)]
	#results = results['IdList']
	#Parallel(n_jobs = 16)(delayed(main_data_collector)(search_text,start = s,end = e) for s,e in zip(xrange(start,end,8),xrange(start + 8,end + 1,8)))
	main_data_collector(search_text, start = start,end = end )
	print 'ending'
#if(__name__ == '__main__')
#main_ops()	
json_load_and_read()
