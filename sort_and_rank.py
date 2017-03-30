#! ~/anaconda2/bin/python
#! -*- coding:utf-8 -*-

import os
import re
import codecs

def read_and_sort(data_path):
	dir_list = os.listdir(data_path)
	data_list = []
	name_ = data_path.split('/')[1]+str(1)
	print name_
	print dir_list
	for each in dir_list:
		with codecs.open(os.path.join(data_path,each)) as data_reader:
			data = data_reader.readlines()
			for datum in data:
				name,val = datum.split('\t')
				data_list.append((name,int(val)))
	data_list = sorted(data_list, key=lambda x:x[1],reverse=True)
	print os.path.join('sorted_output/',name_)
	with codecs.open(os.path.join('sorted_output/',name_),'w+') as data_writer:
		for each in data_list:
			data_writer.write(each[0].strip() +'\t'+str(each[1])+'\n')	
	return

if __name__ == '__main__':
	read_and_sort('output_dir/cardiovascular effects')
