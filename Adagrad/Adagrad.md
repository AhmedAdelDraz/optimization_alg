# **Adagrad**
Adagrad is based on the concept of adaptive **learning rate**. It is built to over come the vanishing gradient for ***sparse*** data -*The data that contains alot of zero values*-. because the update is directly proportional to the gradient. Smaller the gradient smaller the update. but the gradient is directly proportional to the input. Therefore the update is dependent on the input also. 



$ \nabla  \theta_{j} =  \frac{1}{m}  \sum_i^m (h_{ \theta} ( x^{(i)}-y^{(i)})x^{(i)}_{j} $

$v_{t}=v_{t-1}+( \nabla  \theta _{t}  ) ^{2} $

$\theta _{t+1} =  \theta_{t} -  \frac{ \alpha }{  \sqrt{v_{t}} + \varepsilon }  \nabla  \theta _{t}$


$v_{t}$ : accumulates the running sum of square of the gradients.<br>
$\varepsilon$: in the denominator avoids the chances of divide by zero error.

#### **advantage**
* parameters corresponding to sparse features get better updates.

#### **disadvantage**
* the learning rate decay very aggressively as the denominator grows -**not good for parameter corresponding to dense feature**- hence there is no update in value of parameter so learning rate got killed because denominator growing very fast. it reaches to near the minima point but not at the minima.