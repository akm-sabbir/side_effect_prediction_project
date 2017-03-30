#! usr/bin/env python
#! -*- coding -*- utf-8

import os
import sys
import math
import logging
import gensim
import Queue
import re
import Queue
from collections import deque
from collections import defaultdict
from gensim.models import word2vec
from gensim.models import Phrases
sys.path.insert(0,'../')
from gensimWord2vec import load_and_perform_molecular_sim
class Graph(object):
	def __init__(self,net_graph=defaultdict(dict),graph_size = 20):
		"initializes a graph object"
		self.total_edge = 0
		FORMAT = '%(asctime)-15s %(user)-8s %(message)sa'
		logging.basicConfig(filename='networkflow.txt',filemode='w+',format=FORMAT)
		self.logger = logging.getLogger('netflow')
		self._net_graph = net_graph
		self.indegree = [0]*graph_size
		self.outdegree = [0]*graph_size
		return
	def load_model(self,model = None):
		return load_and_perform_molecular_sim(path_name = 'concept_to_vec_1.1')
	def __iter__(self):
		for each in self._net_graph:
			#if(self.outdegree[each] == 0):
				#yield (each,0)
			for neighbor in self._net_graph[each]:
				yield (each,neighbor)
		return
	def vertices(self):
		return list(self._net_graph.keys())
	def check_edges(self,vertex1 = None,vertex2 = None):
		if(vertex1 == None):
			self.logger.info('vertex1 is none which is unexected')
			print('proper vertex is missing')
			return
		if(vertex2 == None):
			self.logger.info('vertex2 is none which is unexpectede')
			print('proper vertex2 is missing')
			return
		if(self._net_graph[vertex1].get(vertex2) != None):
			return 1
		if(self._net_graph[vertex2].get(vertex1) != None):
			return 2
		return 0
	def add_vertex(self,vertex):
		self._net_graph[vertex] = {}
		return
	def get_weight(self, ver1,ver2):
		if(self._net_graph[ver1].get(ver2) == None):
			return 0
		return self._net_graph[ver1][ver2]
	def add_edges(self,ver1,ver2,weight):
		if(ver1 in self._net_graph):
			self._net_graph[ver1][ver2] = weight
		else:
		 	self._net_graph[ver1] = {ver2:weight}
		self.total_edge += 1
		if(ver2 not in self._net_graph):
			self._net_graph[ver2] = {}
		return
	def edges(self):
		
		return
	def _generate_edges(self):
		edges = []
		for node in self._net_graph:
			for neighbor in self._net_graph[node]:
				edges.append((node,neighbor,self._net_graph[node][neighbor]))

				
		return edges
	def augment_path(self,v,s,parent,INF):
	    f = 0 
	    if(v == s):
		    f = INF
		    return f
	    elif(parent[v] != 0):
		    #print(parent[v] + '\n')
		    f = self.augment_path(parent[v],s,parent,min(INF,self._net_graph[parent[v]][v]))
		    self._net_graph[parent[v]][v] -= f
		    if(self._net_graph[v].get(parent[v]) == None):
		        self._net_graph[v][parent[v]] = f
		    else:
		    	self._net_graph[v][parent[v]] += f
	    return f
	def perform_max_flow(self):
		max_flow = 0
		s = 's'
		t = 't'
		parent = defaultdict(str)
		while(True):
			f = 0
			Q = deque()
			dist = defaultdict(int)
			Q.append(s)
			dist[s] = 0
			parent[s] = 0
	 		while(len(Q) != 0):
				u = Q.popleft()
				if(u == t):
					break
				for node in self._net_graph[u]:
					if(self._net_graph[u][node] > 0 and dist[node] == 0):
						dist[node] += dist[u] + 1
						Q.append(node)
						parent[node] = u
			f = self.augment_path(t,s,parent,500000)
		
			if f == 0:
				break
			max_flow += f
			del dist
			del Q
		print  'max flow is %d' % max_flow	
		return max_flow
	def __str__(self):
		string = "vertices: "
		for k in  self._net_graph:
			string += str(k) + " "
		string += "\nedges"
		for edge in self._generate_edges():
			string += str(edge) + " "
		return string
def add_start_node(s,graph,node_list):
    for each in node_list:
    	graph.add_edges(s,each[0],int(each[1]*100))
    return
def add_end_node(t,graph,node_list):
    for each in node_list:
	graph.add_edges(each[0],t,int(each[1]*100))
    return
def mainOps(node_list_a = None, node_list_b = None,edge_weight = None,pair_dist_mat = None):
	
	graph = Graph()
	#for i in xrange(1,11,1):
	#	graph.add_vertex(i)
	#node_list_a = [(1,2,2),(1,3,5),(2,4,1),(2,3,7),('s',1,100),('s',2,100),(3,4,3),(3,'t',5),(4,'t',100)]
	#add_start_node('s',graph,node_list_a)
	#add_end_node('t',graph,node_list_b)
	try:
		concept_model = graph.load_model(model = pair_dist_mat)
	except:
		pass
		print('there is an exception and error here')
	
	for elem in node_list_a:
		for each in node_list_b:
			if(concept_model.similarity(elem[0].lower().strip(),each[0].lower().strip()) > 0.1):
				graph.add_edges(elem[0].lower().strip(),each[0].lower().strip(),int(concept_model.similarity(elem[0].lower().strip(),each[0].lower().strip())*10000))
	for each in node_list_a:
		graph.add_edges('s',each[0].lower().strip(),10000)
	for each in node_list_b:
		graph.add_edges(each[0].lower().strip(),'t',10000)
	#print graph
	#print graph.perform_max_flow()
	f = graph.perform_max_flow()
	return f
	
if __name__ == '__main__':
	pass
	#mainOps()
