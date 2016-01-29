import nltk
from random import randint
import pickle
import numpy as np
#this return the sentences from the from corpus
brown_sents = nltk.corpus.brown.sents()

#i have a problem, the sentences aren't the same size, like the images 
#that have 784 pixels. The sentences vary in length. So, lets get the average
#size of a sentenze and then shape all the other sentences to that size
avg_size = 0
sizes = []
for sent in brown_sents:
	sizes.append(len(sent))
avg = np.mean(sizes)

greater = [] #23801 (20) 43290(10) 49299(7)
less = [] #31794 (20) 12020(10) 6218(7)
equal = [] #1745 (20) 2030(10) 1823(7)
#now, reshape each sentence to 7
def shape(shapeArray, shapeNum):
	brown = []
	for sent in shapeArray:
		if len(sent)<shapeNum:
			#do nothing for now
			dif = shapeNum - len(sent)
			i=0
			new_sent = sent
			while i<dif:
				new_sent += [' ']
				i+=1
			brown += [new_sent]
		elif len(sent)>shapeNum:
			new_sent = sent[:shapeNum]
			brown += [new_sent]
		else:
			brown += [sent]
	return brown

#now i have to break them up into 3 groups, training-validation-test
#len is 57,340, so let's breaking up in the following
#training = 50,000
#validation = 7,340
#test = will be another
#for some reason these are also unicode string
brown = shape(brown_sents, 7)
##but now I have to randomly take out a word and have the answer be that word





training_data = brown[:40000]
validation_data = brown[40000:50000]
test_data = brown[50000:]
