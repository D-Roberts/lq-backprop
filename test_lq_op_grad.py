# MIT License

# Copyright (c) 2020 Denisa A.O. Roberts

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Gradient checks using central differences."""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from lq_op_grad import lq, LqGrad 
tf.compat.v1.disable_eager_execution()


def _extra_feeds(extra_feed_dict, new_feeds):
  """From tensorflow"""
  if not extra_feed_dict:
    return new_feeds
  r = {}
  r.update(extra_feed_dict)
  r.update(new_feeds)
  return r

def numeric_q(full_mat, shape_, dtype_):
  np.random.seed(42)
  a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
  epsilon = np.finfo(dtype_).eps
  delta = 0.1 * epsilon**(1.0 / 3.0)
  sess = tf.compat.v1.Session()
  extra_feed_dict = None
  with sess.as_default():
    x_shape = shape_
    x_size = np.prod(x_shape)
    np.random.seed(42)
    x_init = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
    x_data = x_init
    scale = np.asarray(2 * delta, dtype=dtype_)[()]
    tf_a = tf.constant(a)
    tf_b = lq(tf_a, full_matrices=full_mat)
    y = tf_b[1]
    y_size = np.prod(y.shape)
    x = tf_a
    jacobian_num_q = np.zeros((x_size, y_size), dtype=dtype_)
    for row in range(x_size):
      x_pos = x_data.copy()
      x_neg = x_data.copy()
      x_pos.ravel().view(dtype_)[row] += delta
      y_pos = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_pos}))
      x_neg.ravel().view(dtype_)[row] -= delta
      y_neg = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_neg}))
      diff = (y_pos - y_neg) / scale
      jacobian_num_q[row, :] = diff.ravel().view(dtype_)
  return jacobian_num_q

def theoretic_q(full_mat, shape_, dtype_):
  np.random.seed(42)
  a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
  x_shape = a.shape
  x_size = np.prod(x_shape)
  atf = tf.convert_to_tensor(a)
  l, dy = lq(atf, full_matrices=full_mat)
  dy_size = np.prod(dy.shape)
  jacobian = np.zeros((x_size, dy_size)).astype(dtype_)
  dy_data = np.zeros(dy.shape).astype(dtype_)
  dy_data_flat = dy_data.ravel()
  sess = tf.compat.v1.Session()
  for col in range(dy_size):
    dy_data_flat[col] = 1 
    Q_ = tf.convert_to_tensor(dy_data_flat.reshape(dy.shape))
    with sess.as_default():
      jacobian[:, col] = LqGrad(atf, l, dy, 
        tf.convert_to_tensor(np.zeros(l.shape).astype(dtype_)), Q_).eval().ravel().view(jacobian.dtype)
    dy_data_flat[col] = 0
  return jacobian

def theoretic_l(full_mat, shape_, dtype_):
  np.random.seed(42)
  a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
  x_shape = a.shape
  x_size = np.prod(x_shape)
  atf = tf.convert_to_tensor(a)
  dy, q = lq(atf, full_matrices=full_mat)
  dy_size = np.prod(dy.shape)
  jacobian_tl = np.zeros((x_size, dy_size)).astype(dtype_)
  dy_data = np.zeros(dy.shape).astype(dtype_)
  dy_data_flat = dy_data.ravel()
  sess = tf.compat.v1.Session()
  for col in range(dy_size):
    dy_data_flat[col] = 1
    L_ = tf.convert_to_tensor(dy_data_flat.reshape(dy.shape))
    with sess.as_default():
      jacobian_tl[:, col] = LqGrad(atf, dy, q, L_, 
        tf.convert_to_tensor(np.zeros(q.shape).astype(dtype_))).eval().ravel().view(jacobian_tl.dtype)
    dy_data_flat[col] = 0
  return jacobian_tl

def numeric_l(full_mat, shape_, dtype_):
  np.random.seed(42)
  a = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
  shape_ = a.shape
  epsilon = np.finfo(dtype_).eps
  delta = 0.1 * epsilon**(1.0 / 3.0)
  sess = tf.compat.v1.Session()
  extra_feed_dict = None
  with sess.as_default():
    x_shape = a.shape
    x_size = np.prod(x_shape)
    np.random.seed(42)
    x_init = np.random.uniform(low=-1.0, high=1.0, size=shape_).astype(dtype_)
    x_data = x_init
    scale = np.asarray(2 * delta, dtype=dtype_)[()]
    tf_a = tf.constant(a)
    tf_b = lq(tf_a, full_matrices=full_mat)
    y = tf_b[0]
    y_size = np.prod(y.shape)
    jacobian_num_l = np.zeros((x_size, y_size), dtype=dtype_)
    x = tf_a
    for row in range(x_size):
      x_pos = x_data.copy()
      x_neg = x_data.copy()
      x_pos.ravel().view(dtype_)[row] += delta
      y_pos = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_pos}))
      x_neg.ravel().view(dtype_)[row] -= delta
      y_neg = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_neg}))
      diff = (y_pos - y_neg) / scale
      jacobian_num_l[row, :] = diff.ravel().view(dtype_)
  return jacobian_num_l

def _test_LQ_op(shape, dtypes, full_matrices=False):
  a_np = np.random.uniform(-1, 1, size=shape).astype(dtypes)
  a = tf.convert_to_tensor(a_np)
  l, q = lq(a)
  sess = tf.compat.v1.Session()
  with sess.as_default():
    res = tf.matmul(l, q).eval()
  assert np.allclose(res, a_np)

def test():
    # Unit test cases
  for full_matrices in [False]:
    # Assumes reduced matrices and float32 dtype
    for dtype_ in [np.float32]:
      for rows in 3, 5:
        for cols in 3, 5:
          if rows >= cols or (not full_matrices and rows < cols):
             for batch_dims in [()]: 
              shape = batch_dims + (rows, cols)
              name = "%s_%s_full_%s" % (dtype_.__name__,
                                          "_".join(map(str, shape)),
                                          full_matrices)
              # test LQ op (forward)
              _test_LQ_op(shape, dtype_, full_matrices)

              # test LQ grad (backward)
              if dtype_ == np.float32:
                  rtol, atol = 3e-2, 3e-2
              else:
                  rtol, atol = 1e-6, 1e-6
              assert np.allclose(theoretic_l(full_matrices, shape, dtype_), 
                  numeric_l(full_matrices, shape, dtype_), atol, rtol)
              assert np.allclose(theoretic_q(full_matrices, shape, dtype_), 
                  numeric_q(full_matrices, shape, dtype_), atol, rtol)


if __name__ == '__main__':
  import nose
  nose.runmodule()