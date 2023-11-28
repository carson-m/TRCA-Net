import numpy as np

a = np.array([[1,2,3],[4,5,6]])
b = np.zeros([2,3,3,4])
for i in range(3):
    for j in range(4):
        b[:,:,i,j] = a+i
c = b.transpose([2,3,0,1])
d = c.reshape(12,2,3)
print(d[0,:,:])
print(d[4,:,:])