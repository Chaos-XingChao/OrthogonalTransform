import numpy as np
import sys

def InitMatrix(dim):
#	A = np.random.rand(dim,dim)
#	return tuningMatrix(A)
	return np.eye(dim)


def evaluteTrain(X,A,Y):
	#try to evalute sum(x_i.T * A * y_i)
	sum = 0.0
	for i in range(X.shape[0]):
		tempX = X[i].reshape(1,X.shape[1])
		sum += np.dot(tempX ,np.dot(A,Y[:,i:i+1]))
	return sum[0][0]
#	return sum[0][0] / X.shape[0]

def train(X, A, Y, alpha=0.25, max_iter=-1, error_rate=1e-4, min_show=100):
	#train the matrix A when optimize min( sum(x_i.T * A * y_i) )
	old_state = []
	iter = 1
	start_alpha = alpha
	while True:
#		if (iter % min_show == 0):
#			sys.stderr.write("\balpha is : " + str(alpha) + " iteration is : " + str(iter) + " performance is : " + str(old_state[-1]))
#			sys.stderr.flush()

		if (iter == max_iter):
			break
		if (alpha < start_alpha * 1e-4):
			alpha = start_alpha * 1e-4
		iter = iter + 1
		if not old_state:
			trainStep(X,A,Y, alpha)
			T = tuningMatrix(A)
			if T != "":
				A = T
				old_state = [X,A,Y,alpha,iter,evaluteTrain(X,A,Y)]
			else:
				alpha = alpha * 2 / 3			
#				sys.stderr.write("learning rate too lager. Tuning...\n\talpha : " + str(alpha) + "\n")
		else:
			trainStep(X,A,Y, alpha)
			T = tuningMatrix(A)
			if T != "":
				evaluate_value = evaluteTrain(X,T,Y)
				if evaluate_value > old_state[-1]:
					A = T
					if evaluate_value - old_state[-1] < error_rate:
						break
					old_state = [X,A,Y,alpha,iter,evaluate_value]
				else:
					alpha = alpha * 2 / 3
#					sys.stderr.write("learning rate too lager. The gradient value loss.. Tuning...\n\talpha : " + str(alpha) + "\n")
			else:
				alpha = alpha * 2 / 3	
#				sys.stderr.write("learning rate too lager. Tuning...\n\talpha : " + str(alpha) + "\n")
#		sys.stderr.write("performance is : " + str(old_state[-1]) + "\n")
#		sys.stderr.flush()
	return A	

def trainStep(X, A, Y, alpha=0.25):
	#for one step, may change alpha ==> learning rate
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			A[i][j] = A[i][j] + alpha * np.dot(X[:,i],Y[j,:])
	return A

def tuningMatrix(A):
	#find nearest orthogonal matrix using simplest method.
	if (A.T * A == np.eye(A.shape[0])).all():
		return A	
	try:
		U,e,V = np.linalg.svd(A)
		B = np.dot(U,np.dot(np.eye(e.shape[0]), V))
		return B
	except:
		return ""

