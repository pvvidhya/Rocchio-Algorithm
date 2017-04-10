from __future__ import division
from collections import Counter
import math
import re
import os
import collections
import time

class index:
	def __init__(self,path):
		self.path= path

	def read_doc(self): 
		f= open('TIME.ALL', "r")
		co= f.read()
		token= re.sub('\W+',' ', co)
		token= token.split()
		token = [x for x in token if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

		####reading STOP words####
		f= open('TIME.STP', "r")
		co= f.read()
		token_stp= re.sub('\W+',' ', co)
		token_stp= token_stp.split()

		####removing STOP words from document####
		token_terms = []
		for terms in token:
			if terms not in token_stp:
				token_terms.append(terms)

		with open('TIME.ALL') as f:
			file_name = []
			for line in f:
				if '*TEXT' in line:
					file_name.append(line[1:9])
			
		uid = []
		for i in range(len(file_name)):
			fi= i
			uid.append(fi)
			i = i+1
		# global uidict
		uidict= {}
		uidict = dict(zip(file_name,uid))

		with open('TIME.ALL') as f:
			buf = f.read()

		li_docs = []
		x = 0
		line = buf.splitlines()
		for m in (buf.splitlines()):
			if (len(m.strip()) != 0):
				if m[0:5] == "*TEXT" or m[0:5] == "*STOP":
					if x != 0:
						li_docs.append(wr+ ' ')
					wr = ""
					x = 1
				else:
					if len(wr) == 0:
						wr = m	
					else:
						wr = wr+ ' '+ m
		global docs
		docs = []
		for x in li_docs:
			t = re.sub('\W+',' ', x)
			t = t.split()
			docs.append(t)
		return docs

	def dic_count(docs):

		#####Document frequenies######
		global dic_count
		dic_count = {}
		for x in docs:
			so_list = []
			for term in x:
				if term not in dic_count.keys():
					dic_count[term] = 1.0
				else:
					if term not in so_list:	
						dic_count[term] += 1.0
				so_list.append(term)
		return dic_count

	def dic(docs):
		# #####doc_id and tf weights#######
		global dic 
		dic={}
		for x in range(len(docs)):
		    for n in (docs[x]):
		        if n not in dic.keys():
		            dic[n]=[[x+1,1.0]]
		        else:
		            if (x+1) not in [b[0] for b in [a for a in dic[n]]]:
		                dic[n].append([x + 1,1.0])
		            else:
		                for a in range (len(dic[n])):
		                    if (dic[n][a][0]) == x + 1:
		                        dic[n][a][1] += 1
		return dic

	def tf_idf_norm(dic, dic_count):
		# #####tf-idf vector form######
		n_docs = 423
		global tf_idf_norm
		tf_idf_norm={}
		for m in dic.keys():
		    for n in dic[m]:
		        if n[0] not in tf_idf_norm.keys():
		            tf_idf_norm[n[0]]= ((1 + math.log10(n[1])) * math.log10(n_docs / dic_count[m]) )**2 
		        else:
		            tf_idf_norm[n[0]]+= ((1 + math.log10(n[1])) * math.log10(n_docs / dic_count[m]) )**2  

		for m in tf_idf_norm.keys():
		    tf_idf_norm[m]=math.sqrt(tf_idf_norm[m])
		return (tf_idf_norm)

	####read query#####
	def query(self):
		with open('TIME.QUE') as f:
			buf = f.read()
		global dic_query
		dic_query = {}
		wrq =''
		l = 0
		line = buf.splitlines()
		for m in (buf.splitlines()):
			if (len(m.strip()) != 0):
				if (m[0:5]) == '*FIND':
					k = (m.split()[1])
					if l != 0:
						dic_query[int(k)-1] = wrq + ' ' 
					wrq = ''
				elif (m[0:5]) != '*STOP':
					if wrq !='':
						wrq = wrq + ' ' + m
					else:
						wrq = wrq+m
					l = 1
		dic_query[int(k)] = wrq + ' '

		####Tokenizing the query###
		q_terms = re.sub('\W+',' ', buf)
		q_terms = q_terms.split()
		# print (len(q_terms))

		####reading STOP words####
		f= open('TIME.STP', "r")
		co= f.read()
		token_stp= re.sub('\W+',' ', co)
		token_stp= token_stp.split()
			
		####removing STOP words from query####
		global query_terms
		query_terms = []
		for terms in q_terms:
			if terms not in token_stp:
				query_terms.append(terms)
		return query_terms

	def query_vector(query_terms, dic_count):	
	# #####Query idf####
		global query_vector
		query_vector = {}
		for x in query_terms:
			if x in dic_count.keys():    
				query_vector[x] = math.log10(float(len(docs))/float((dic_count[x])))
		# print query_vector
		return query_vector

	def cosine(query_vector, dic, docs, dic_count, tf_idf_norm):
	####Cosine Similarity####
		global cosine_dict
		cosine_dict = {}
		for query in query_vector.keys():
		    for dlist in dic[query]:
		        if dlist[0] in cosine_dict.keys():
		            cosine_dict[dlist[0]] += ((1+math.log10(dlist[1]))*math.log10(len(docs)/dic_count[query])*query_vector[query])/tf_idf_norm[dlist[0]]
		        else:
		            cosine_dict[dlist[0]] = ((1+math.log10(dlist[1]))*math.log10(len(docs)/dic_count[query])*query_vector[query])/tf_idf_norm[dlist[0]]
		# print (cosine_dict)
		return cosine_dict

	def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma, new_query_vector):

		p = positive_fb.split() 
		p = [int(x) for x in p]

		n = negative_fb.split() 
		n = [int(x) for x in n]

		global roc_dic_q
		roc_dic_q = {}
		if new_query_vector == Null:
			for query in query_vector.keys():
				roc_dic_q[query] = alpha*query_vector[query]
		else:
			for query in new_query_vector.keys():
				roc_dic_q[query] = alpha*new_query_vector[query]

		global dic_new_p
		dic_new = {}
		for x in p:
			for k,v in dic.items():
				for item in v:
					if item[0] == x:
						dic_new[k] = item

		global roc_dic_p
		roc_dic_p = {}
		for k,v in dic_new_p.items():
			if k not in roc_dic_p.keys():
				roc_dic_p[k] = ((((1+math.log10(v[1]))*math.log10(len(docs)/dic_count[k]))/tf_idf_norm[v[0]])/len(p))
		
		for key in roc_dic_p:    
			roc_dic_p[key] *=  beta
		
		global dic_new_n
		dic_new_n = {}
		for x in n:
			for k,v in dic.items():
				for item in v:
					if item[0] == x:
						dic_new_n[k] = item
		
		global roc_dic_n
		roc_dic_n = {}
		for k,v in dic_new.items():
			if k not in roc_dic_n.keys():

				roc_dic_n[k] = ((((1+math.log10(v[1]))*math.log10(len(docs)/dic_count[k]))/tf_idf_norm[v[0]])/len(n))
		
		for key in roc_dic_n:    
			roc_dic_n[key] *= gamma
		

		#####ROCCHIO ALGORITHM#####
		A = Counter(roc_dic_q)
		B = Counter(roc_dic_p)
		C = Counter(roc_dic_n)
		global Query_roc
		Query_roc = []
		for i in range(5):
			Q = A + B - C
			new_query_vector = dict(Q)
		return new_query_vector

	####Read relevent documents####
	def read_relevant(self):
		with open('TIME.REL') as f:
			buf = f.read()

		global dic_relevant
		dic_relevant = {}
		line = buf.splitlines()
		for m in (buf.splitlines()):
			if(len(m.strip()) != 0):
				k = (m.split())
				lis_relevant = []
				for j in range(1,len(k)):
					lis_relevant.append(int(k[j]))
				dic_relevant[int(k[0])] = lis_relevant
		return (dic_relevant)

	def retrieved_docs(cosine_dict,k):
		retrieved = []
		sorted_cos = sorted(cosine_dict, key=cosine_dict.get, reverse=True)
		for key in sorted_cos[:k]:
			retrieved.append(key)
		return ('RETRIEVED: %s' % retrieved)

	def precision(retrieved, relevant):
		###Precision####
		pre = []
		for m in retrieved:
			if m in relevant:
				pre.append(m)

		Precision = ((len(pre))/(len(retrieved)))
		print ("Precision: "), (Precision)

def main():
	path = "/Users/Vidhy/Desktop/COURSES/IR/IR_ASG3/time"
	ind= index(path)
	total_docs = ind.read_doc()
	dic_id = ind.dic()
	dic_count = ind.dic_count(docs)
	tf_idf_norm = tf_idf_norm(dic, dic_count)
	query_terms = ind.query(self)
	query_vector = query_vector(query_terms, dic_count)
	cosine_dict = cosine(query_vector, dic, docs, dic_count, tf_idf_norm)
	alpha = 1.0
	beta = 0.75
	gamma = 0.15
	print ("enter the positive documents")
	pos_feedback = raw_input()
	print ("enter the negative documents")
	neg_feedback = raw_input()
	k = raw_input("Number of retrieved documents")
	k = int(k)
	new_query_vector = {}
	new_query_vector = rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma, new_query_vector)
	retrieved = retrieved_docs(cosine_dict,k)
	for query_id, query_text in dic_query.items(): 
		relevant = sorted(list(dic_relevant[query_id]))
	precision = precision(retrieved, relevant)
	print ("Query text: %s" %query_text)
	print ("Query ID: %d" %query_id) 


if __name__ == '__main__':
	main()