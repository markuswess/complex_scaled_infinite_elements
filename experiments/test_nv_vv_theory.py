import numpy as np

a = np.array([[1,2,3]]).T
b = np.array([[0,2,1]]).T
n = np.array([[1,8,2]]).T
v = np.array([[1,2,1]]).T
#a = np.array([[1,3]]).T
#b = np.array([[0,1]]).T
#n = np.array([[1,4]]).T
#v = np.array([[1,1]]).T

d=a.shape[0]
n = n/np.linalg.norm(n)
proj = np.eye(d)-n@n.T
print(a.T@proj@((n.T@v)**2*np.eye(d)+v@v.T)@proj@b)
print((v.T@v)*(a.T@proj@b))
