#!usr/bin/env python
#! -*- coding:utf:8 -*-

import os
import sys
import math
import nltk
import logging
from collections import defaultdict
import numpy as np
class Node(object):
	def __init__(self,indegree = 0,outdegree = 0,pointIn = None):
		self.score = 0
		self.indegree = indegree
		self.outdegree = outdegree
		self.PR = np.random.random_sample()
		self.total_weight = 0
		self.weighted_PR = np.random.random_sample()
		self.pointIn = pointIn
		return
	def set_score(self,score = 0.1):
		self.score = score
	def __iter__(self):
		return
	def __repr__(self):
		return
	def rank():
		return
		
