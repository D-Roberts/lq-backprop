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

# ==============================================================================


"""Differentiable LQ decomposition of any matrix order."""

import numpy as np 
import tensorflow as tf


def lq(a, full_matrices=False):
    """a is a tensorflow tensor.
    """
    res = tf.linalg.qr(tf.linalg.adjoint(a), full_matrices=full_matrices)
    return (tf.linalg.adjoint(res[1]), tf.linalg.adjoint(res[0]))


def _LqGrad(l, q, dl, dq):

    def copyltu(M):
        # shape of M is [batch, m, m]
        eye =tf.eye(M.shape[-1])
        lower = tf.linalg.band_part(M, -1, 0) - eye * M
        lower_mask = tf.linalg.band_part(tf.ones(M.shape), -1, 0)
        ret = lower_mask * M + tf.linalg.adjoint(lower)
        return ret
    
    def _LqGradSquareAndWideMatrices(l, q, dl, dq):
        l_t = tf.linalg.adjoint(l)
        q_t = tf.linalg.adjoint(q)
        M = tf.matmul(l_t, dl) - tf.matmul(dq, q_t)
        return tf.linalg.triangular_solve(
                l, dq + tf.matmul(copyltu(M), q), lower=True, adjoint=True)
    
    m, n = l.shape.dims[-2], q.shape.dims[-1].value
    
    if m <= n:
        return _LqGradSquareAndWideMatrices(l, q, dl, dq)
    
    # partition l, a for deep a
    a = tf.matmul(l, q)
    y = a[..., n:, :]
    u = l[..., :n, :]
    dv = dl[..., n:, :]
    du = dl[..., :n, :]
    dy = tf.matmul(dv, q)
    dx = _LqGradSquareAndWideMatrices(u, q, du,
                                        dq + tf.matmul(dv, y, adjoint_a = True, adjoint_b = False))
    return tf.concat([dx, dy], axis=-2)
