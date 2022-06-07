import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import math

def compile():
	yale = sio.loadmat('yalefaces.mat')
	yalefaces = yale['yalefaces']
	fig,ax = plt.subplots()
	dataset = []
	for i in range(0,yalefaces.shape[2]):
		x=yalefaces[:,:,i]
		#ax.imshow(x, extent=[0,1,0,1])
		#plt.imshow(x, cmap=plt.get_cmap('gray'))
		#time.sleep(0.1)
		#plt.show()
		img = np.array(x)
		img = img.reshape(img.shape[0]*img.shape[1])
		dataset.append(img)
	dataset = np.array(dataset)
	return dataset

def normalize(dataset):
	mean = np.mean(dataset,axis=0)
	dataset= dataset - mean
	return dataset, mean

def eigen(dataset):
	cov_mat = np.dot(dataset.T, dataset)
	eig_val, eig_vect = np.linalg.eig(cov_mat)
	#eig_vect = np.dot(eig_vect,dataset.T)
	#eig_vect=eig_vect.T
	#for i in range(eig_vect.shape[1]):
		#eig_vect[:,i]=eig_vect[:,i]/np.linalg.norm(eig_vect[:,i])
	#eig_vect=eig_vect.T
	return eig_val.astype(float), eig_vect.astype(float)

def pca(eig_val,eig_vect,k):
	k_eig_val = eig_val.argsort()[::-1][-k:]
	eig_faces = []
	for i in k_eig_val:
		eig_faces.append(eig_vect[:,i])
	eig_faces = np.array(eig_faces).T
	return eig_faces

def variation(eig_val,threshold):
	sum_vals = np.sum(eig_val)
	for k in range(len(eig_val)):
		sum_vals_k=np.sum(eig_val[:k])
		var=sum_vals_k/sum_vals
		if var>threshold:
			break
	return k
		
dataset=compile()
num_components = 2016
dataset,mean=normalize(dataset)
eig_val, eig_vect=eigen(dataset)
eig_val_sorted =  np.sort(eig_val)[::-1]
k_min_95=variation(eig_val_sorted,0.95)
k_min_99=variation(eig_val_sorted,0.99)
print('Number of principal components to represent 95% variation:')
print(k_min_95)
print('Percentage reduction in dimension for 95% representation:')
print(1-k_min_95/len(eig_val))
print('Number of principal components to represent 99% variation:')
print(k_min_99)
print('Percentage reduction in dimension for 99% representation:')
print(1-k_min_99/len(eig_val))
plt.semilogy(eig_val_sorted[:])
plt.show()
plt.clf()
eig_faces = pca(eig_val,eig_vect,num_components)
#eig_faces=eig_faces.T
reshaped_faces = []
for i in range(20):
	reshaped_faces.append(eig_faces[:,i].reshape(48,42))
reshaped_faces=np.array(reshaped_faces)


fig, axs = plt.subplots(4, 5, sharex=True, sharey=True)
for i in range(20):
	index_2=int(i%5)
	if i==0:
		index_1=0
		index_2=0
		axs[index_1, index_2].imshow(mean.reshape(48,42), cmap=plt.get_cmap('gray'))
	else:
		index_1=math.floor(i/5)
		axs[index_1, index_2].imshow(reshaped_faces[i], cmap=plt.get_cmap('gray'))
	#plt.show()
	axs[index_1, index_2].set_title('Eig face #'+str(i))
plt.show()



