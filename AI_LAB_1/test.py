from numpy import loadtxt

data = loadtxt('ex1data1.txt', delimiter=',')
print data[:,0]