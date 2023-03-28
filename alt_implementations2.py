# Implementation from GPT4

import six
import re
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
from six.moves import range
from tensorflow.keras import initializers
from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.python.framework import smart_cond as tf_utils

from qkeras import BaseQuantizer, stochastic_round
from qkeras.quantizers import _get_scaling_axis, _get_scale
from pprint import pprint

from timing_utils import get_time_info


class quantized_bits3(BaseQuantizer):

    ALPHA_OPTIONS = ("auto", "auto_po2")

    def __init__(
        self,
        bits=8,
        integer=0,
        symmetric=False,
        keep_negative=True,
        qnoise_factor=1.0,
        use_stochastic_rounding=False,
        use_ste=True,
        alpha=None,
        scale_axis=None,
    ):

        super().__init__()

        self.bits = K.cast_to_floatx(bits)
        self.integer = K.cast_to_floatx(integer)
        self.symmetric = K.cast_to_floatx(symmetric)
        self.keep_negative = K.cast_to_floatx(keep_negative)
        self.qnoise_factor = K.cast_to_floatx(qnoise_factor)
        self.use_stochastic_rounding = tf.cast(use_stochastic_rounding, tf.bool)
        self.use_ste = tf.cast(use_ste, tf.bool)
        self.alpha = K.cast_to_floatx(1.0) if alpha is None else alpha
        self.scale_axis = scale_axis
        self._check_bits()

        if isinstance(self.alpha, six.string_types):
            self._check_str_alpha()

    @tf.function
    def __call__(self, x):
        x = K.cast_to_floatx(x)

        unsigned_bits = self.bits - self.keep_negative
        integer_repr_scale = tf.math.reciprocal(
            K.cast_to_floatx(tf.bitwise.left_shift(1, unsigned_bits - self.integer))
        )

        xq = tf.case(
            [
                (tf.equal(unsigned_bits, 0.0), lambda: self._sign_function(x)),
                (
                    tf.cast(isinstance(self.alpha, six.string_types), tf.bool),
                    lambda: self._auto_alpha_computation(
                        x, unsigned_bits, integer_repr_scale
                    ),
                ),
            ],
            default=lambda: self._multi_bit_computation(
                x, unsigned_bits, integer_repr_scale
            ),
        )

        return self._quantized_return(x, xq)

    def _auto_alpha_computation(self, x, unsigned_bits, base_integer_repr_scale):
        """Compute quantized value for automatically determined alpha"""

        int_repr_min, int_repr_max = self._int_repr_bounds(unsigned_bits)

        def _autoscale(x, alpha_scale, int_repr_min, int_repr_max):
            int_xq = self._get_quantized_integer(
                x, alpha_scale, int_repr_min, int_repr_max
            )
            return alpha_scale, int_xq

        # Get the minimum floating point scale that does not clip the max of x
        alpha_scale = self._get_alpha_scale(x, int_repr_max, int_repr_min)
        conditions = [
            (
                tf.cast(self.alpha == "auto_po2", tf.bool),
                lambda: self._po2_autoscale(x, alpha_scale, int_repr_min, int_repr_max),
            ),
            (
                tf.cast(self.alpha == "auto", tf.bool),
                lambda: _autoscale(x, alpha_scale, int_repr_min, int_repr_max),
            ),
        ]

        alpha_scale, int_xq = tf.case(conditions, exclusive=True)

        xq = int_xq * alpha_scale
        alpha_scale = alpha_scale / base_integer_repr_scale

        self.scale = alpha_scale

        return xq

    def _sign_function(self, x):
        if isinstance(self.alpha, six.string_types):
            self.alpha = 1.0

        self._set_numeric_alpha()

        neg_res = K.cast_to_floatx(-self.keep_negative)
        nonneg_res = K.cast_to_floatx(1)

        xq = tf.where(tf.math.greater_equal(x, 0), nonneg_res, neg_res)
        return xq * self.alpha

    def _multi_bit_computation(self, x, unsigned_bits, integer_repr_scale):
        self._set_numeric_alpha()

        int_repr_min, int_repr_max = self._int_repr_bounds(unsigned_bits)
        int_xq = self._get_quantized_integer(
            x, integer_repr_scale, int_repr_min, int_repr_max
        )
        xq = int_xq * integer_repr_scale

        return xq * self.alpha

    def _get_quantized_integer(self, x, integer_repr_scale, int_repr_min, int_repr_max):
        scaled_x = x / integer_repr_scale
        clipped_scaled_x = K.clip(scaled_x, int_repr_min, int_repr_max)
        int_xq = _round_through(
            clipped_scaled_x,
            use_stochastic_rounding=self.use_stochastic_rounding,
            precision=1.0,
        )

        return int_xq

    def _set_numeric_alpha(self):
        self.alpha = K.cast_to_floatx(self.alpha)
        self.scale = self.alpha

    def _get_alpha_scale(self, x, int_repr_max, int_repr_min):
        """Get the minimum floating point scale that does not clip the max of x"""

        axis = self._get_axis(x)

        levels = int_repr_max - int_repr_min
        conditions = [
            (
                tf.cast(self.keep_negative, tf.bool),
                lambda: (K.max(tf.math.abs(x), axis=axis, keepdims=True) * 2) / levels,
            ),
            (
                tf.cast(not self.keep_negative, tf.bool),
                lambda: K.max(x, axis=axis, keepdims=True) / levels,
            ),
        ]

        alpha_scale = tf.case(conditions, exclusive=True)
        return tf.math.maximum(alpha_scale, K.epsilon())

    def _get_axis(self, x):
        """Get axis for alpha scale computation"""

        len_axis = len(x.shape)
        if len_axis != 1:
            axis = _get_scaling_axis(self.scale_axis, len_axis)
        else:
            axis = [0]
        return axis

    def _po2_autoscale(self, x, alpha_scale, int_repr_min, int_repr_max):
        alpha_scale = K.pow(
            2.0, tf.math.round(K.log(alpha_scale + K.epsilon()) / np.log(2.0))
        )

        # TODO: set up proper tensorflow loop
        def loop_body(_, alpha_scale, __, iteration_count):
            """Loop body for least squares autoscaling"""

            int_xq = self._get_quantized_integer(
                x, alpha_scale, int_repr_min, int_repr_max
            )
            new_alpha_scale = _get_scale(
                alpha="auto_po2", x=x, q=int_xq, scale_axis=self.scale_axis
            )
            return alpha_scale, new_alpha_scale, int_xq, iteration_count + 1

        def loop_cond(last_alpha_scale, alpha_scale, __, iteration_count):
            """Loop condition for least squares autoscaling"""

            tensors_equal = tf.math.reduce_all(tf.equal(last_alpha_scale, alpha_scale))
            no_more_iterations = tf.math.greater_equal(iteration_count, 5)
            return tf.math.logical_or(tensors_equal, no_more_iterations)

        for _ in range(5):
            last_alpha_scale = alpha_scale
            int_xq = self._get_quantized_integer(
                x, alpha_scale, int_repr_min, int_repr_max
            )
            alpha_scale = _get_scale(
                alpha="auto_po2", x=x, q=int_xq, scale_axis=self.scale_axis
            )
            if tf.math.reduce_all(tf.equal(last_alpha_scale, alpha_scale)):
                break
        return alpha_scale, int_xq

    def _int_repr_bounds(self, unsigned_bits):
        unsigned_bits_po2 = K.cast_to_floatx(tf.bitwise.left_shift(1, unsigned_bits))
        int_repr_min = self.keep_negative * (-unsigned_bits_po2 + self.symmetric)
        int_repr_max = unsigned_bits_po2 - 1

        return int_repr_min, int_repr_max

    def _quantized_return(self, x, xq):
        return tf.cond(
            self.use_ste,
            lambda: x + tf.stop_gradient(self.qnoise_factor * (-x + xq)),
            lambda: (1 - self.qnoise_factor) * x
            + tf.stop_gradient(self.qnoise_factor * xq),
        )

    def _check_bits(self):
        if self.bits < self.integer + self.keep_negative:
            err_msg = (
                f"Bit count {self.bits} must exceed {self.integer + self.keep_negative}"
            )
            raise ValueError(err_msg)

    def _check_str_alpha(self):
        if not self.alpha in self.ALPHA_OPTIONS:
            raise ValueError(
                f"Invalid alpha '{self.alpha}' for auto alpha computation. "
                f"Must be one of {self.ALPHA_OPTIONS}"
            )


def _round_through(x, use_stochastic_rounding=False, precision=0.5):
    """Rounds x but using straight through estimator.

    We use the trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182).

    Straight through estimator is a biased estimator for the rounding
    operation defined by Hinton"s Coursera Lecture 9c where dL/dx is made
    equal to dL/dy for y = f(x) during gradient computation, where f(x) is
    a non-derivable function. In that case, we assume df/dx = 1 in:

    dL   dL df   dL
    -- = -- -- = --
    dx   df dx   dy

    (https://www.youtube.com/watch?v=LN0xtUuJsEI&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9&index=41)

    Arguments:
      x: tensor to perform round operation with straight through gradient.
      use_stochastic_rounding: if true, we perform stochastic rounding.
      precision: by default we will use 0.5 as precision, but that can overriden
        by the user.

    Returns:
      Rounded tensor.
    """
    conditions = [
        (
            use_stochastic_rounding,
            lambda: tf_utils.smart_cond(
                K.learning_phase(),
                lambda: x + tf.stop_gradient(-x + stochastic_round(x, precision)),
                lambda: x + tf.stop_gradient(-x + tf.round(x)),
            ),
        ),
        (
            tf.ones_like(use_stochastic_rounding, dtype=tf.bool),
            lambda: x + tf.stop_gradient(-x + tf.round(x)),
        ),
    ]

    output = tf.case(conditions, exclusive=True)
    return output


get_time_info(__file__)
