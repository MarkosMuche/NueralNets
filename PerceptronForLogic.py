
import numpy as np
and_=np.array([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,1.]])
x =np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y=np.array([0.,1.,1.,1.])
weights=np.array([0.,0.])
b=0.;
func = lambda pred: 0 if pred < 0 else 1
for i in range(10):
    for X,Y in zip(x,y):
       pred = np.dot(weights,X) + b
       yhat=func(pred)
       errr = Y - yhat
       weights += 0.1*errr*X
       b += 0.1*errr
print(weights)
print(b)

def predict(x):
    pred = np.dot(weights, x) + b
    yhat = func(pred)
    return yhat
print(predict([0,0]))
print(predict([0,1]))
print(predict([1,0]))
print(predict([1,0]))

