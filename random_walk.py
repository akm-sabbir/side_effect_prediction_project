#! usr/bin/env python
#! -*- coding:utf-8 -*-
import os
import sys
import math
import nltk
import logging
from gensim.models import word2vec
from gensim.models import Phrases
import gensim.models.doc2vec
from Node import Node
sys.path.insert(0,'../')
from collections import defaultdict
from gensimWord2vec import load_and_perform_molecular_sim

class pair(object):
	def __init__(self,a = None, b = None):
		self.a = a
		self.b = b
		return

class Graph(object):
	def __init__(self,graph_dict={},damping = 0.85,isDirected = True):
		"initializes a graph object"		
		self.damping = damping
		self.total_edge = 0
		FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
		logging.basicConfig(filename = 'logger.txt',filemode = 'w+',format = FORMAT)
		self.logger = logging.getLogger('graph_walk')
		self._graph_dict = graph_dict
		self.isDirected = isDirected
		self.indegree = defaultdict(int)
		self.outdegree = defaultdict(int)
		self.indegree_list = defaultdict(list)
		self.iteration = 0
		return
	def __iter__(self):
		# Only those nodes which has neighbors are gonna be yields
		# as a result nodes with zero in degree will have no participation in calculation
		for each in self._graph_dict:
			for neighbor in self._graph_dict[each][0]:
				yield (each,neighbor)
		return

	def load_model(self):
		return load_and_perform_molecular_sim(path_name='../concept_to_vec_1.1')
	def vertices(self):
		return list(self._graph_dict.keys())
	def edges(self):
		return self._generate_edges()
	def add_vertex(self,vertex):
		if vertex not in self._graph_dict:
			self._graph_dict[vertex] =[{},0]
		return
	def check_edges(self,vertex1,vertex2):
		if(isDirected == True):
			if(self._graph_dict[vertex1][0].get(vertex2) != None):
				return True
		else:
			if(self._graph_dict[vertex1][0].get(vertex2) != None or self._graph_dict[vertex2][0][vertex1] != None):
				return True
		return False
	def return_weight(self, vertex1,vertex2):
		weight = 0
		if(self_graph_dict[vertex1][0].get(vertex2) != None):
			weight = self._graph_dict[vertex1][0][vertex2]
		return weight
	def load_concept_model(self,path_name = None):
		if(path_name == None):
			self.logger.info('problem is %s','path need to be existed for model')
			return

		return
	def add_edge(self,*edge ):
		(vertex1,vertex2,weight,directed) = edge
		if(vertex1 in self._graph_dict):
			self._graph_dict[vertex1][0][vertex2] = weight
			#self._graph_dict[vertex][1] = weight
		else:
			self._graph_dict[vertex1] = [{vertex2:weight},0]
		if(directed == False):
			if(vertex2 in self._graph_dict):
				self._graph_dict[vertex2][0][vertex1] = weight
			else:
			 	self._graph_dict[vertex2] = [{vertex1:weight},0]
		self.total_edge += 1

		return
	def _detect_cycle(self, source_node = None, des_node = None):
		return False
	def _generate_edges(self):
		edges = []
		for node in self._graph_dict:
			for neighbor in self._graph_dict[node][0]:
				edges.append((node,neighbor,self._graph_dict[node][0][neighbor]))
		return edges
	def _generate_indegree_outdegree(self):
		for node in self._graph_dict:
			self.outdegree[node] += len(self._graph_dict[node][0])
		for node in self._graph_dict:
			for neighbor in self._graph_dict[node][0]:
				self.indegree[neighbor] += 1
				self.indegree_list[neighbor].append(node)
		return
	def init_rank(self):
		for node in self._graph_dict:
			self._graph_dict[node][1] = Node(indegree = self.indegree[node], outdegree = self.outdegree[node],pointIn = self.indegree_list)
		return
	def calculate_page_rank(self):
		for node in self._graph_dict:
			new_pr = 0
			for each in self.indegree_list[node]:
				new_pr = new_pr + (self.damping*(self._graph_dict[each][1].PR/self._graph_dict[each][1].outdegree))
			new_pr += (1-self.damping)
			self._graph_dict[node][1].PR = new_pr
		return
	def calculate_weighted_page_rank(self):
		for node in self._graph_dict:
			for each in self.indegree_list[node]:
				self._graph_dict[node][1].total_weight += self._graph_dict[each][0][node]
		for node in self._graph_dict:
			self._graph_dict[node][1].weighted_PR = 1 - self.damping
			for each in self.indegree_list[node]:
				self._graph_dict[node][1].weighted_PR = self._graph_dict[node][1].weighted_PR + (self.damping*self._graph_dict[node][1].weighted_PR*self._graph_dict[each][0][node]/self._graph_dict[node][1].total_weight)
		return
	def _total_edge(self):
		return sum([self.outdegree[node] for node in self._graph_dict])

	def __str__(self):
		res = "vertices"
		for k in self._graph_dict:
			res += str(k) +" "
		res += "\nedges: "
		for edge in self._generate_edges():
			res += str(edge) + " "
		return res
def dependency(s_node,des_node,accumulation,total_cnt):
	return float(float(accumulation[s_node][des_node])/float(total_cnt[s_node]))

def centrality_score(g_new = None):
	if(g_new == None):
		print('There is something wrong in the graph\n')
		return 0
	cnt = Counter()
	for ind,node in enumerate(g_new):
		for ind_n,neighbor in enumerate(g_new[node]):
			if(g_new[node].get(neighbor) != None):
				cnt[node] += g_new[node][neighbor]		
			
	return cnt

def build_graph(g = None,label_list = None,concept_net = None,total_cnt = None,cosine = 0):
	g_new = Graph()
	node_tracker = {}
	threshold = 7
	model = g_new.load_model()
	g_new.logger.info('model is loaded %s', model)
	for i in xrange(0,len(g),1):
		for j in xrange(i+1,len(g),1):
			if(j - i >= threshold):
				break
			for each in label_list[g[i]]:
				for datum in label_list[g[j]]:
					if(cosine == 0):
						if(concept_net[each].get(datum) != None):
							weight = dependency(each,datum,concept_net,total_cnt)
							#if(node_tracker.get(each) == None):
							#	node_tracker[each] = g[i]
							#if(node_tracker.get(datum) == None):
							#	node_tracker[datum] = g[j]
							#print('adding new edge')
							g_new.add_edge(each,datum,weight,True)
					else:
						try:
							if(model[each.lower()] != None and model[datum.lower()] != None):
							
								if(model.similarity(each.lower(),datum.lower())>0.10):
									weight = model.similarity(each.lower(),datum.lower())
									g_new.add_edge(each,datum,weight,True)
						except:
							pass
						#print('adding new edges')
					if(node_tracker.get(each) == None):
						node_tracker[each] = g[i]
					if(node_tracker.get(datum) == None):
						node_tracker[datum] = g[j]

	return (g_new,node_tracker)
	'''					
	g = {"a":({"d":0},0),
	     "b":({"c":0},0),
	     "c":({"b":0,"c":0,"d":0,"e":0},0),
	     "d":({"a":0,"c":0},0),
	     "e":({"c":0},0),
	     "f":({},0)}
	graph = Graph(g)
	
	print("Vertices of graph:")
	print(graph.vertices())
	print("Edges of graph:")
	print(graph.edges())
	
	print("Add vertex:")
	graph.add_vertex("z")

	print("Vertices of graph:")
	print(graph.vertices())
	 
	print("Add an edge:")
	graph.add_edge({"a","z"})
		    
	print("Vertices of graph:")
	print(graph.vertices())

	print("Edges of graph:")
	print(graph.edges())

	print('Adding an edge {"x","y"} with new vertices:')
	graph.add_edge({"x","y"})
	print("Vertices of graph:")
	print(graph.vertices())
	print("Edges of graph:")
	print(graph.edges())
	'''
def _read_data(path_name = None,file_name = None):
	 file_list = os.listdir(path_name)
	 dict_ = {}
	 for each in file_list:
		if(each.find('unique') == -1):
			continue
		with open(os.path.join(path_name,each)) as data_reader:
			data = data_reader.readlines()
			for datum in data:
				(ver1,ver2) = datum.split('\t')[3:5]
				if(dict_.get(ver1)== None):
					dict_[ver1] = [ver2]
				else:
				 	dict_[ver1].append(ver2)
				if(dict_.get(ver2) == None):
					dict_[ver2] = [ver1]
				else:
				 	dict_[ver2].append(ver1)
	 build_graph(dict_)
		 
	 return 
def graph_operation():

	return
if(__name__ == "__main__"):
	pass
	#read_data(path_name = 'dataset/mrrel')
	#build_graph()
