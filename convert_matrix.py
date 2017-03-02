import itertools
import numpy as np
from numpy import linalg as LA
from nltk import ngrams
import urllib2
import urllib
from bs4 import BeautifulSoup
import re
import requests
import justext
import unicodedata

inp = open('oldpage_ml.txt').readlines()

dict_cols = {}
dict_rows = {}
global_2d = []
#global_2d.append([])
queries = []
for i in range(0,len(inp)):
	queries.append( inp[i].rstrip())
        
links = []
c = 0
#print len(queries)
for i in range(0,len(queries)):
	temp = queries[i].split(':')
	dict_rows[i] = temp[0]

	if (temp[1] != ''):
		links = temp[1].split(',')        
		for j in range(0,len(links)):
			dict_cols[c] = links[j]
			c = c+1	       
			#print c 
		global_2d.append([0]*c + [1] * len(links))
	else:
		global_2d.append([])
	


cols = map(len, global_2d)
#print cols
max_len = max(cols)

for i in range(0,len(cols)):
	if (cols[i]<max_len):
		diff = max_len-cols[i]
		#print diff
                temp = global_2d[i]
                change = temp+[0]*diff
		#print len(change)
		global_2d[i] = change 
                #print global_2d[i]
                
samples = np.array(global_2d)
cols = map(len, global_2d)
#print cols
#print np.shape(samples)

top = open('Topics_big/Topics_ml.txt').readlines()
topics = []
for i in range(0,len(inp)):
	topics.append( top[i].rstrip())

ti = 245
for t in topics:
	dict_cols[t] = t
	ti = ti+1

z = np.zeros((186,1067), dtype='int64')
#z = np.zeros(len(queries),len(top))
#print z
samples = np.append(samples, z, 1)
print np.shape(samples)

mean_vector = np.array([])
temp = np.array([])
for i in range(0,186):          
	mean = np.mean(samples[i,:])
	temp = np.append(temp, [mean])
	#print temp
        mean_vector = np.row_stack(temp)
print('Mean Vector:\n', len(mean_vector))

scatter_matrix = np.zeros((186,186))
for i in range(samples.shape[1]):
    	scatter_matrix += (samples[:,i].reshape(186,1) - mean_vector).dot((samples[:,i].reshape(186,1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
print "values of scatter matrix"
print eig_val_sc
print "vectors of scatter matrix"
print eig_vec_sc

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
	print 'decreasing order:'
    	print(i[0])

matrix_w = np.hstack((eig_pairs[0][1].reshape(186,1), eig_pairs[1][1].reshape(186,1)))
projected = matrix_w.T.dot(samples)
predicted = matrix_w.dot(projected)
gain = samples - predicted
print 'Reconstruction Gain:\n',gain


