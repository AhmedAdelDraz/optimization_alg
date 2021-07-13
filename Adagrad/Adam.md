# Adam
So far in Adagrad, RMSProp we were calculating different learning rates for different parameters, can we have different momentums for different parameters Adam algorithm introduces the concept of **adaptive momentum** along with **adaptive learning** rate.

Adaptive Moment Estimation in Adam **$m_{t}$** computes the exponentially decaying average of previous gradients along with an adaptive learning rate. 

Adam is a combined form of Momentum-based GD and RMSProp. In Momentum-based GD, previous gradients -*history*- are used to compute the current gradient whereas, in RMSProp previous gradients -*history*- are used to adjust the learning rate based on the features.

### **Learning updates**
### $ \nabla  \theta_{j} =  \frac{1}{m}  \sum_i^m (h_{ \theta} ( x^{(i)}-y^{(i)})x^{(i)}_{j} $

### $m_{t}=\beta_1 m_{t-1}+(1-\beta_1)( \nabla  \theta _{t}  ) $

### $v_{t}=\beta_2 v_{t-1}+(1-\beta_2)( \nabla  \theta _{t}  ) ^{2} $

### **Bias Correction terms**

* ## $m_{t}=\frac{m_{t}}{1-\beta_1^{t}}$

* ## $v_{t}=\frac{v_{t}}{1-\beta_2^{t}}$

## $\theta _{t+1} =  \theta_{t} -  \frac{ \alpha }{  \sqrt{v_{t}} + \varepsilon }  \nabla  \theta _{t}$

$v_{t}$: is exponentially decaying average of all the previous squared gradients.
