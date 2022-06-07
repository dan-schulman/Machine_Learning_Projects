import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy.io as sio
import math

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))
    
def iterate(theta_iter, X, Y):
	z = np.dot(theta_iter[1:].T,X[1:,:])+theta_iter[0] #theta*x
	lamda = 10
	n = sigmoid(z)
	D=np.diag(n[0,:]*(1-n[0,:])) #create diagonal entries of the sigmoid functions to calculate hessian
	gradient_lamda=np.diag([2*lamda]*785) #create regularization as diagonal matrix
	gradient = np.dot(X, (n-Y).T)+2*theta_iter*lamda #G=x*(n-y)
	#hessian = np.dot(np.dot(np.dot(X,X.T),n),1-n)+2*lamda
	hessian = np.matmul(np.matmul(X,D),X.T)+gradient_lamda #H=X*D*X'
	cost = -1.0*np.sum(Y*np.log(n)+(1.0-Y)*np.log(1.0-n))+lamda*np.linalg.norm(theta_iter) #-l(\theta)
	theta_new = theta_iter - np.dot(np.asarray(np.linalg.inv(np.mat(hessian))),gradient) #iteration over theta
	return theta_new,cost
	
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')
x = mnist_49_3000['x']
y = mnist_49_3000['y']
d, n = x.shape

for i in range(3000):
	if y[0,i]==-1:
		y[0,i]=0
	
i = 1  # Index of the image to be visualized
plt.imshow(np.reshape(x[:, i], (int(np.sqrt(d)), int(np.sqrt(d)))))
plt.show()

x = np.insert(x, 0, 1, axis=0)
x_train=np.array(x[:,:2000])
y_train=y[:,:2000]
x_test=np.array(x[:,2000:])
y_test=y[:,2000:]

theta = np.zeros((785,1)) #1 dimension extra than X for b
criteria=1 #stopping criteria
cost_old=1 #holding variables for cost
cost_new=1
e=0.000001
counter_iterations=0
while criteria>e:
	cost_old=cost_new
	theta,cost_new = iterate(theta,x_train,y_train) #run the iterative algorithm
	criteria=abs(cost_old-cost_new)/cost_old
	counter_iterations+=1
	print(criteria)

print('Number of iterations = ', counter_iterations,'\n')
print('Objective function final value = ', cost_new,'\n')

most_confident=np.zeros((20,2)) #0 is index, 1 is confidence. keep track of most confident guesses
label = np.zeros((1,1000)) #classification labels for test set
confidence = np.zeros((1,1000)) #quantify the confidence. To be assigned to most_confident
for i in range(1000):
	t=np.dot(theta.T,x_test[:,i])
	classif=sigmoid(t) #find sigmoid for test cases
	confidence[0,i]=(classif-0.5)*(classif-0.5) #confidence is defined as how far the classif. is from threshold 0.5
	if classif>0.5: #check threshold of classification
		label[0,i]=1
	else:
		label[0,i]=0

correct=0
incorrect=0
for i in range(1000):
	most_confident=most_confident[most_confident[:,1].argsort()] #sorting list to keep track of most confident guesses
	if label[0,i]==y_test[0,i]:
		correct+=1
	else:
		incorrect+=1
		for j in range(20):
			if confidence[0,i]>most_confident[j,1]: #only check incorrect confident guesses
				most_confident[j,1]=confidence[0,i]
				most_confident[j,0]=i
				break
error = incorrect/1000 #test error
x_test = np.delete(x_test, 0, axis=0)
fig, axs = plt.subplots(4, 5, sharex=True, sharey=True)
for i in range(20):
	index_2=int(i%5-1)
	if i%5==0:
		index_1=int(i/5-1)
	else:
		index_1=math.floor(i/5)
	axs[index_1, index_2].imshow(np.reshape(x_test[:, int(most_confident[i,0])], (int(np.sqrt(d)), int(np.sqrt(d)))))
	#plt.show()
	axs[index_1, index_2].set_title('True label ='+str(y_test[0,int(most_confident[i,0])]))
	print('True label = ', y_test[0,int(most_confident[i,0])])
plt.show()
print('Test error = ', error)
