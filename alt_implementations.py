"""Alternative implementations of qkeras quantizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from qkeras.quantizers import _get_scale
from pprint import pprint

from timing_utils import get_time_info
import sys


class quantized_bits2(BaseQuantizer):
    """Linear quantization with fixed number of bits.

    This quantizer maps values to the nearest value of a fixed number of
    outputs that are evenly spaced, with possible scaling and stochastic
    rounding.

    The core computation is:
        1. Scale the tensor by a factor
        2. Clip the tensor to a specified range
        3. Round to the nearest integer (the "integer representation")
        4. Scale back by 1/factor

    This "integer representation" range is determined by
        - The number of bits we have to represent the number
        - Whether we want to have a symmetric range or not
        - Whether we want to keep negative numbers or not

    The scale is defined by either the user or the quantizer itself.
    # TODO: Add more details about the scale.

    The quantizer also supports a number of other optional features:
        - Stochastic rounding
        - Quantization noise
        - Straight-through estimator activation

    Example:
        ```python
        # 8-bit quantization with 3 integer bits
        q = quantized_bits(8, 3)
        x = tf.constant([0.0, 0.5, 1.0, 1.5, 2.0])
        q(x)
        # tf.Tensor([0. 0. 1. 2. 2.], shape=(5,), dtype=float32)
        ```
    Note:
        The computation differs very slightly from the above description for
        1-bit symmetric quantization where we keep negative numbers,
        since the above formula would map -1 to -1, 0 to 0, and 1 to 1
        (thus representing three numbers with one bit). In this case, the
        quantizer is just a sign function, where the sign of zero is positive.

    Args:
        bits (int): Number of bits to represent the number.
        integer (int): Number of bits to the left of the decimal point.
        symmetric (bool): If true, we will have the same number of values for positive
            and negative numbers.
        alpha (tensor, str, None): The scaling factor per channel. If None, the scaling factor
            is 1 for all channels. If "auto", scaling factor is calculated as
            the minimum floating point scale that does not clip the max of x.
            if "auto_po2", the scaling factor is chosen as the power of two that
            minimizes squared error between the scaled quantized x and the original x.
        keep_negative (bool): If true, we do not clip negative numbers.
        use_stochastic_rounding (bool): If true, we perform stochastic rounding.
        scale_axis (int): Which axis to calculate scale from.
        qnoise_factor (float): A scalar from 0 to 1 that represents the level of
            quantization noise to add. This controls the amount of the quantization
            noise to add to the outputs by changing the weighted sum of
            (1 - qnoise_factor) * unquantized_x + qnoise_factor * quantized_x.
        var_name (str or None): A variable name shared between the tf.Variables
            created in the build function. If None, it is generated automatically.
        use_ste (bool): Whether to use "straight-through estimator" (STE) method or
            not. (Reference: https://arxiv.org/pdf/1903.05662.pdf)
        use_variables (bool): Whether to make the quantizer variables to be dynamic
            tf.Variables or not.

    Returns:
        function: Function that computes fixed-point quantization with bits.

    Raises:
        ValueError:
            - If bits is not positive, or is too small to represent integer.
            - If integer is negative.
            - If alpha is a string but not one of ("auto", "auto_po2").

    """

    # alpha enum- need for uniform typing of alpha for auto-alpha options
    AUTO_ALPHA_ENUM = K.cast_to_floatx(1.0)
    AUTO_PO2_ALPHA_ENUM = K.cast_to_floatx(2.0)

    # immutable map of alpha strings to alpha enum
    ALPHA_STRING_TO_ENUM = (
        ("auto", AUTO_ALPHA_ENUM),
        ("auto_po2", AUTO_PO2_ALPHA_ENUM),
    )

    def __init__(
        self,
        bits=8,
        integer=0,
        symmetric=0,
        keep_negative=True,
        alpha=None,
        use_stochastic_rounding=False,
        scale_axis=None,
        qnoise_factor=1.0,
        var_name=None,
        use_ste=True,
        use_variables=False,
    ):
        super().__init__()

        # Set _initialized parameter to False to prevent the setters from
        # performing preliminary calculations
        self._initialized = False
        self.bits = bits
        self.integer = integer
        self.symmetric = symmetric
        self.keep_negative = keep_negative
        self.qnoise_factor = qnoise_factor
        self.use_stochastic_rounding = use_stochastic_rounding
        self.use_ste = use_ste
        self.alpha = alpha
        self.scale_axis = scale_axis
        self.var_name = var_name
        self.use_variables = use_variables
        # set scale as a tf.Variable so that it can be updated
        # within tf.functions
        self.scale = tf.Variable(1.0, name="scale", shape=tf.TensorShape(None))

        # Perform preliminary calculations based on attributes above
        self._initialized = True
        self._calc_input_independent_attributes()

        # Auto-scaling not needed for sign quantization
        # TODO: make sure this is needed
        if self.auto_alpha and self.use_sign_function:
            self.auto_alpha = tf.cast(False, tf.bool)
            self.alpha = K.cast_to_floatx(1.0)

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, bits):
        if bits <= 0:
            raise ValueError(f"Bit count {bits} must be positive")
        self._bits = K.cast_to_floatx(bits)
        if self._initialized:
            self._calc_input_independent_attributes()

    @property
    def integer(self):
        return self._integer

    @integer.setter
    def integer(self, integer):
        if integer < 0:
            raise ValueError(f"Integer bit count {integer} must be nonnegative")
        self._integer = K.cast_to_floatx(integer)
        if self._initialized:
            self._calc_input_independent_attributes()

    @property
    def symmetric(self):
        return self._symmetric

    @symmetric.setter
    def symmetric(self, symmetric):
        self._symmetric = K.cast_to_floatx(symmetric)
        if self._initialized:
            self._calc_input_independent_attributes()

    @property
    def keep_negative(self):
        return self._keep_negative

    @keep_negative.setter
    def keep_negative(self, keep_negative):
        self._keep_negative = K.cast_to_floatx(keep_negative)
        if self._initialized:
            self._calc_input_independent_attributes()

    @property
    def qnoise_factor(self):
        return self._qnoise_factor

    @qnoise_factor.setter
    def qnoise_factor(self, qnoise_factor):
        self._qnoise_factor = K.cast_to_floatx(qnoise_factor)

    @property
    def use_stochastic_rounding(self):
        return self._use_stochastic_rounding

    @use_stochastic_rounding.setter
    def use_stochastic_rounding(self, use_stochastic_rounding):
        self._use_stochastic_rounding = tf.cast(use_stochastic_rounding, tf.bool)

    @property
    def use_ste(self):
        return self._use_ste

    @use_ste.setter
    def use_ste(self, use_ste):
        self._use_ste = tf.cast(use_ste, tf.bool)

    @property
    def scale_axis(self):
        return self._scale_axis

    @scale_axis.setter
    def scale_axis(self, scale_axis):
        self._scale_axis = scale_axis

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """
        Set alpha and auto_alpha attributes, and check if alpha is valid
        Also, set scale if not auto_alpha.

        Alpha can be passed in as a string (e.g. "auto", indicating a
        data-dependent alpha) or a tensor. In order to avoid typing issues with
        the construction of tensorflow graphs, we want the represent the alpha
        using uniform data types. Therefore, we convert the string to an enum (tensor),
        and then store a boolean auto_alpha to indicate whether the alpha is dependent
        on the data or not.
        """

        if alpha is None:
            self._alpha = K.cast_to_floatx(1.0)
            self.auto_alpha = tf.cast(False, tf.bool)
        elif isinstance(alpha, six.string_types):
            self._check_str_alpha(alpha)
            self._alpha = dict(self.ALPHA_STRING_TO_ENUM)[alpha]
            self.auto_alpha = tf.cast(True, tf.bool)
        else:
            self._alpha = K.cast_to_floatx(alpha)
            self.auto_alpha = tf.cast(False, tf.bool)

    def _check_str_alpha(self, alpha):
        """Check the quantizer has been given a valid alpha string"""

        alpha_options = dict(self.ALPHA_STRING_TO_ENUM).keys()
        if not alpha in alpha_options:
            raise ValueError(
                f"Invalid alpha '{alpha}' for auto alpha computation. "
                f"Must be one of {alpha_options}"
            )

    def _calc_input_independent_attributes(self):
        """Calculate and set attributes that are independent of __call__ input"""
        assert (
            self._initialized
        ), "Must initialize before calling _calc_input_independent_attributes"
        self._set_unsigned_bits()
        self._set_sign_function()
        self._set_integer_repr_scale()
        self._set_int_repr_bounds()

    def _set_unsigned_bits(self):
        """Compute unsigned bits scale (for integer representation), and store as attribute"""

        self.unsigned_bits = self.bits - self.keep_negative
        if self.unsigned_bits < self.integer:
            err_msg = (
                f"Bit count {self.bits} must exceed {self.integer + self.keep_negative}"
            )
            raise ValueError(err_msg)

    def _set_sign_function(self):
        """Set boolean based on whether to use sign function"""

        self.use_sign_function = tf.cast(self.unsigned_bits == 0, tf.bool)

    def _set_integer_repr_scale(self):
        """Get scale for integer representation, as given by parameters other than alpha
        Store as an attribute"""
        integer_repr_scale = tf.math.reciprocal(
            K.cast_to_floatx(
                tf.bitwise.left_shift(1, self.unsigned_bits - self.integer)
            )
        )
        self.integer_repr_scale = integer_repr_scale

    def _set_int_repr_bounds(self):
        """Get bounds of rounded integer representation and set as attributes"""

        unsigned_bits_po2 = K.cast_to_floatx(
            tf.bitwise.left_shift(1, self.unsigned_bits)
        )
        int_repr_min = self.keep_negative * (-unsigned_bits_po2 + self.symmetric)
        int_repr_max = unsigned_bits_po2 - 1

        self.int_repr_min = int_repr_min
        self.int_repr_max = int_repr_max
        self.levels = self.int_repr_max - self.int_repr_min

    @tf.function
    def __call__(self, x):
        """Core quantization function"""
        # build if not done so already
        self._build()

        # Data type conversion
        x = K.cast_to_floatx(x)

        xq = tf.cond(
            self.use_sign_function,
            lambda: self._sign_function(x),
            lambda: self._multi_bit_computation(x),
        )

        return self._quantized_return(x, xq)

    def _multi_bit_computation(self, x):
        """Quantization multi-bit representation- differs for auto and static alpha"""

        xq = tf.cond(
            self.auto_alpha,
            lambda: self._auto_alpha_computation(x),
            lambda: self._static_alpha_computation(x),
        )

        return xq

    def _auto_alpha_computation(self, x):
        """Compute quantized value for automatically determined alpha"""

        # Get the minimum floating point scale that does not clip the max of x
        alpha_scale = self._get_alpha_scale(x)

        # Autoscale functions for tf.cond below
        def autoscale():
            """Quantize with `alpha_scale` above"""
            int_xq = self._get_quantized_integer(x, alpha_scale)
            return alpha_scale, int_xq * alpha_scale

        def po2_autoscale():
            """Get the "best" po2 scale for the data"""
            _alpha_scale, int_xq = self._po2_autoscale(x, alpha_scale)
            return _alpha_scale, int_xq * _alpha_scale

        alpha_scale, xq = tf.case(
            [
                (tf.equal(self.alpha, self.AUTO_ALPHA_ENUM), autoscale),
                (tf.equal(self.alpha, self.AUTO_PO2_ALPHA_ENUM), po2_autoscale),
            ],
        )

        self.scale.assign(alpha_scale / self.integer_repr_scale)

        return xq

    def _sign_function(self, x):
        """Sign indicator function for 1-bit quantization"""

        neg_res = K.cast_to_floatx(-self.keep_negative)
        nonneg_res = K.cast_to_floatx(1)

        xq = tf.where(tf.math.greater_equal(x, 0), nonneg_res, neg_res)

        self.scale.assign(self.alpha)

        return xq * self.alpha

    def _static_alpha_computation(self, x):
        """Compute quantized value for multi-bit quantization with static alpha"""

        int_xq = self._get_quantized_integer(x, self.integer_repr_scale)
        xq = int_xq * self.integer_repr_scale

        self.scale.assign(self.alpha)

        return xq * self.alpha

    def _get_quantized_integer(self, x, integer_repr_scale):
        """Get x quantized in integer representation"""

        scaled_x = x / integer_repr_scale
        clipped_scaled_x = K.clip(scaled_x, self.int_repr_min, self.int_repr_max)
        int_xq = _round_through(
            clipped_scaled_x,
            use_stochastic_rounding=self.use_stochastic_rounding,
            precision=1.0,
        )

        return int_xq

    def _get_alpha_scale(self, x):
        """Get the minimum floating point scale that does not clip the max of x"""

        axis = self._get_axis(x)

        def alpha_scale_keep_negative():
            """Get alpha scale when keeping negative values"""

            return (K.max(tf.math.abs(x), axis=axis, keepdims=True) * 2) / self.levels

        def alpha_scale_no_negative():
            """Get alpha scale when dropping negative values"""

            return K.max(x, axis=axis, keepdims=True) / self.levels

        alpha_scale = tf.cond(
            tf.equal(self.keep_negative, 1.0),
            alpha_scale_keep_negative,
            alpha_scale_no_negative,
        )

        return tf.math.maximum(alpha_scale, K.epsilon())

    def _get_axis(self, x):
        """Get axis for alpha scale computation"""

        len_axis = tf.rank(x)
        axis = tf.cond(
            tf.not_equal(len_axis, 1),
            lambda: _get_scaling_axis(self.scale_axis, len_axis),
            lambda: tf.convert_to_tensor([0]),
        )
        return axis

    def _po2_autoscale(self, x, alpha_scale):
        """Get an approximation of the "best" po2 scale using least squares"""

        alpha_scale = K.pow(
            2.0, tf.math.round(K.log(alpha_scale + K.epsilon()) / np.log(2.0))
        )

        def loop_body(_, alpha_scale, __):
            """Loop body for least squares autoscaling"""

            int_xq = self._get_quantized_integer(x, alpha_scale)
            new_alpha_scale = _get_scale(
                alpha="auto_po2",
                x=x,
                q=int_xq,
                scale_axis=self.scale_axis,
            )
            return alpha_scale, new_alpha_scale, int_xq

        def loop_cond(last_alpha_scale, alpha_scale, __):
            """Loop condition for least squares autoscaling- stop when the scale
            converges or after 5 iterations"""

            tensors_not_equal = tf.math.reduce_any(
                tf.not_equal(last_alpha_scale, alpha_scale)
            )
            return tensors_not_equal

        # Need a tensor of the same shape as alpha_scale that is not equal to alpha_scale
        dummy_alpha_scale = -tf.ones_like(alpha_scale)

        _, alpha_scale, int_xq = tf.while_loop(
            loop_cond,
            loop_body,
            (dummy_alpha_scale, alpha_scale, x),
            shape_invariants=(
                alpha_scale.shape,
                alpha_scale.shape,
                tf.TensorShape(None),
            ),
            maximum_iterations=5,
        )  # x and dummy_alpha_scale not used as inputs, just needed to determine shapes

        return alpha_scale, int_xq

    def _quantized_return(self, x, xq):
        """Return quantized value, incorporating noise and gradient stopping and ste"""

        return tf.cond(
            self.use_ste,
            lambda: x + tf.stop_gradient(self.qnoise_factor * (-x + xq)),
            lambda: (1 - self.qnoise_factor) * x
            + tf.stop_gradient(self.qnoise_factor * xq),
        )

    def _build(self):
        """Build the quantizer if not built yet."""
        if not self.built:
            self.build(var_name=self.var_name, use_variables=self.use_variables)


def _get_scaling_axis(scale_axis, len_axis):
    """Get the axis to perform auto scaling with"""

    if not scale_axis is None:
        axis = tf.range(scale_axis)
        axis = tf.concat([axis, tf.range(scale_axis + 1, len_axis)], axis=0)
    else:
        if K.image_data_format() == "channels_last":
            axis = tf.range(tf.math.maximum(len_axis - 1, 0))
        else:
            axis = tf.range(1, len_axis)
    return axis


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
