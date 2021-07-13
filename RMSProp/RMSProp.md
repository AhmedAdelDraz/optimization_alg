# **RMSProp**
RMSProp is designed to overcomes the decaying learning rate problem of adagrad and prevents the rapid growth in $v_{t}$ so gradient in dense feature doesn't vanish.
Instead of accumulating squared gradients from the beginning, it accumulates the previous gradients in some portion **weighted manner**.


### **Learning updates**
$ \nabla  \theta_{j} =  \frac{1}{m}  \sum_i^m (h_{ \theta} ( x^{(i)}-y^{(i)})x^{(i)}_{j} $

$v_{t}=\beta v_{t-1}+(1-\beta)( \nabla  \theta _{t}  ) ^{2} $

$\theta _{t+1} =  \theta_{t} -  \frac{ \alpha }{  \sqrt{v_{t}} + \varepsilon }  \nabla  \theta _{t}$

$v_{t}$: is exponentially decaying average of all the previous squared gradients.

**Exponentially Weighted Moving Average**

$v_{t}=\beta v_{t-1}+(1-\beta)T_{t} $

RMSProp concerns with adaptive learning rate. However, it suffers from a large number of oscillations with high learning rate or large gradient.