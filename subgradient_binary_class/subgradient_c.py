import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

def shuffle_array(x,y):
	#shuffle columns of data
	x=x[:,:20000]
	y=y[:,:20000]
	np.random.seed() #new random seed every time
	np.random.seed(np.random.randint(0,20000))
	y_new=np.zeros((1,np.size(y)))
	x1=x[1,:]
	x2=x[2,:]
	y=y[0,:]
	s = np.arange(x1.shape[0]) #same shuffle shape for x and y
	np.random.shuffle(s)
	x1=x1[s]
	x2=x2[s]
	y_new[0,:]=y[s]
	x0=np.ones(20000)
	x_new=np.stack((x0, x1,x2)) #reformat
	return x_new,y_new

np.random.seed(0)
nuclear = sio.loadmat('nuclear.mat')

x = nuclear['x']
y = nuclear['y']

x = np.insert(x, 0, 1, axis=0)
y = np.asarray(y)

lamda=0.001
theta=np.zeros(3) #A is st [0 w]=A*theta
A=np.zeros((3,3))
A[1,1]=1
A[2,2]=1
iterations=70
J_array=np.zeros(iterations)
iter_array=np.zeros(iterations)
theta_final=0

for j in range(iterations):	
	#shuffle columns
	[x,y]=shuffle_array(x,y)
	J=0
	gradient=np.zeros(3)
	for i in range(20000):		
		aj=100/(j+1) #learning rate
		if int(np.dot(y[:,i],np.dot(theta,x[:,i])))>=1:
			gradient=np.zeros(3) #ui=0
			J+=0
		else:
			gradient=-1*y[:,i]*x[:,i] #ui=-y.x
			J+=(1-int(np.dot(y[:,i],np.dot(theta,x[:,i]))))
		Ji_gradient=gradient/20000+(lamda)*np.dot(A,theta)/20000
		theta=theta-aj*Ji_gradient #update theta using gradient of Ji
	J=J/20000
	J+=(lamda/2)*np.linalg.norm(np.dot(A,theta))
	J_array[j]=J
	iter_array[j]=j
	print(j)
	if j==(iterations-1):
		theta_final=theta

x=np.delete(x,0,0)
negInd = y == -1
posInd = y == 1
x_line = np.linspace(0,8,100)
y_line=(-x_line*theta[1]-theta[0])/theta[2]
plt.plot(x_line,y_line,'-k')
plt.scatter(x[0, negInd[0, :]], x[1, negInd[0, :]], color='b')
plt.scatter(x[0, posInd[0, :]], x[1, posInd[0, :]], color='r')
plt.figure(1)
plt.show()


plt.plot(iter_array*20000,J_array,'-k')
plt.xlabel('iterations')
plt.ylabel('J')
plt.show()

print('Min objective function=',J_array[(iterations-1)])
print('Theta [b w1 w2]=',theta)

