import numpy as np
from copy import deepcopy

z = np.genfromtxt('spambase.data', dtype=float, delimiter=',')
np.random.seed(0)  # Seed the random number generator
rp = np.random.permutation(z.shape[0])  # random permutation of indices
z = z[rp, :]  # shuffle the rows of z
x = z[:, :-1]
y = z[:, -1]
x_copy=deepcopy(x)
x_test=x[2001:,:]
y_test=y[2001:]
x_train = x[:2000,:]
y_train = y[:2000]

median_array=np.median(x_train,axis=0);

#initialize variables
num_spam = 0;
num_ham = 0;
word_counts_spam = [0]*57
word_counts_ham = [0]*57
word_prob_spam = [0]*57
word_prob_ham = [0]*57
y_class = [0]*np.size(x_test)
likelihood_spam = [0]*np.size(x_test)

for i in range(2000):
	for j in range(57):	
		if x_train[i,j]>median_array[j]: #> yields better results than >= since a lot of the medians have value 0
			x_train[i,j]=2; #the word occurs often
			if y_train[i]==1:
				word_counts_spam[j]+=1 #keep track of which words occur in which class
			else:
				word_counts_ham[j]+=1
		else:
			x_train[i,j]=1; #word does not occur often
			
	if y_train[i]==1:
		num_spam=num_spam+1
	elif y_train[i]==0:
		num_ham=num_ham+1

spam_prob=num_spam/np.size(y_train) #overall probability of spam
ham_prob=num_ham/np.size(y_train) #higher probability. Majority classification accuracy ~0.6. Accuracy should be above this

correct_class = 0
incorrect_class = 0

for i in range(57):
	word_prob_spam[i] = (word_counts_spam[i])/num_spam #probability that given a spam, the email contains a word 
	word_prob_ham[i] = (word_counts_ham[i])/num_ham #probability that given a ham, the email contains a word

for i in range(np.size(y_test)):
	PS=1 #these are initialization variables for the multiplication of conditional probabilities that given a label, a word is contained
	PH=1
	for j in range(57):
		if x_test[i,j]>median_array[j]: 
			x_test[i,j]=2 #the word occurs often	
		else:
			x_test[i,j]=1
			
		if x_test[i,j]==2: #calculate pmfs for words that occur often
			PS=PS*word_prob_spam[j]
			PH=PH*word_prob_ham[j]
			
	likelihood_spam[i] = spam_prob*PS/(spam_prob*PS+ham_prob*PH) #overal Bayes likelihood of spam		
	if likelihood_spam[i]>0.5: #threshold for binary classification
		y_class[i]=1
	else:
		y_class[i]=0
	
	if y_class[i]==y_test[i]:
		correct_class+=1
	else:
		incorrect_class+=1

accuracy = correct_class/(correct_class+incorrect_class)

print('test error for majority class = ',spam_prob,'\n')		
print('test error = ',1-accuracy,'\n')
