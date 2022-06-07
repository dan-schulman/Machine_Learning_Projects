import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
np.random.seed(0)
nuclear = sio.loadmat('nuclear.mat')

x = nuclear['x']
y = nuclear['y']

negInd = y == -1
posInd = y == 1

#insert one to account for b
x = np.insert(x, 0, 1, axis=0)
y = np.asarray(y)

lamda=0.001
theta=np.zeros(3)
A=np.zeros((3,3))
#A matrix is st [0 w]=A*theta
A[1,1]=1
A[2,2]=1

iterations=70 #stopping criteria
J_array=np.zeros(iterations)
iter_array=np.zeros(iterations)
theta_final=0
for j in range(iterations):
	gradient=np.zeros(3)
	J=0
	for i in range(20000):
		#max (0, 1-y*theta*x)
		if int(np.dot(y[:,i],np.dot(theta,x[:,i])))>=1:
			gradient+=np.zeros(3) #ui=0
			J+=0
		else:
			gradient-=y[:,i]*x[:,i] #ui=-y.x
			J+=(1-int(np.dot(y[:,i],np.dot(theta,x[:,i]))))	
	#add regularization
	gradient+=lamda*np.dot(A,theta) 
	aj=100/(j+1) #learning rate
	theta=theta-aj*gradient/20000 #iterate theta
	J=J/20000
	J+=(lamda/2)*np.linalg.norm(np.dot(A,theta)) #update J with normalization
	J_array[j]=J #save values of J
	iter_array[j]=j
	if j==(iterations-1):
		theta_final=theta

x=np.delete(x,0,0)
x_line = np.linspace(0,8,100)
y_line=(-x_line*theta[1]-theta[0])/theta[2] #plot estimated line
plt.plot(x_line,y_line,'-k')
plt.scatter(x[0, negInd[0, :]], x[1, negInd[0, :]], color='b')
plt.scatter(x[0, posInd[0, :]], x[1, posInd[0, :]], color='r')
plt.figure(1)
plt.show()


plt.plot(iter_array,J_array,'-k')
plt.xlabel('iterations')
plt.ylabel('J')
plt.show()

print('Min objective function=',J_array[69])
print('Theta [b w1 w2]=',theta)
