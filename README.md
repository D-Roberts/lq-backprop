# LQ Matrix Backpropagation Algorithm Implementation

TensorFlow implementation of differentiable LQ matrix decomposition for square, wide and deep tensors.

# To Use
Requirements: tf >v1; Python > 3.6.

Recommended: install Anaconda. 
Create a tensorflow environment.

```
# tf cpu only; v2 by default at this time.

conda create -n tf tensorflow
conda activate tf
# conda install numpy
git clone https://github.com/D-Roberts/lq_backprop.git
cd lq_backprop

# to run tests
conda install nose # new terminal may be necessary
nosetests -v test_lq_op_grad.py 

# to use

# Example:

import tensorflow as tf 
import numpy as np 
from lq_op_grad import lq, LqGrad

np.random.seed(42)
a_np = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
a = tf.convert_to_tensor(a_np)
l, q = lq(a)
grad = LqGrad(a, l, q, tf.ones_like(l), tf.ones_like(q))
print(grad)
```

# To Cite

If you use this implementation in your research please cite (coming soon).