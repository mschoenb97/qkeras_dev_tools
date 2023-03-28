"""Tests for correctness of the alt implementations compared to qkeras"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from functools import lru_cache
from itertools import product
import random
import sys
import traceback

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from timing_utils import get_time_info

tf.random.set_seed(0)
random.seed(0)

QUANTIZED_BITS_PARAMS = {
    "alpha": (None, "auto", "auto_po2", 2),
    "bits": (1, 4, 8),
    "integer": (0, 1),
    "symmetric": (True, False),
    "keep_negative": (True, False),
    "qnoise_factor": (1.0, 0.5, 0.0),
    "use_stochastic_rounding": (True, False),
    "use_ste": (True, False),
}

TEST_X_VALUES = (
    0,
    *np.linspace(-2, 2, 50).tolist(),
    tf.random.uniform((10,)),
    tf.random.normal((10, 10)),
)


def quantized_bits_grid_accuracy_tests(alt_quantized_bits, max_test_count=None):

    param_grid = list(product(*QUANTIZED_BITS_PARAMS.values()))
    random.shuffle(param_grid)

    if max_test_count is not None:
        param_grid = param_grid[:max_test_count]

    for params in tqdm(param_grid, desc="Accuracy tests (grid)"):

        kwargs = dict(zip(QUANTIZED_BITS_PARAMS.keys(), params))

        _check_quantized_bits_correctness(alt_quantized_bits, kwargs)

    print("\nGrid accuracy tests passed!\n")


def quantized_bits_linear_accuracy_tests(alt_quantized_bits):
    """Test that alt_quantized_bits and qkeras.quantized_bits
    return the same result for all params in QUANTIZED_BITS_PARAMS,
    trying only one param change at a time"""

    # extra kwargs for special cases
    extra_kwargs_list = [
        {"alpha": "auto", "symmetric": True, "keep_negative": True},
        {"alpha": "auto_po2", "symmetric": True, "keep_negative": True},
        {"alpha": "auto", "symmetric": True, "keep_negative": True, "integer": 2},
        {"alpha": "auto_po2", "symmetric": True, "keep_negative": True, "integer": 2},
    ]

    linear_kwargs_list = []

    for param_name, param_values in QUANTIZED_BITS_PARAMS.items():
        for param_value in param_values:
            kwargs = {param_name: param_value}
            linear_kwargs_list.append(kwargs)

    kwargs_list = linear_kwargs_list + extra_kwargs_list

    for kwargs in tqdm(kwargs_list, desc="Accuracy tests (linear)"):
        _check_quantized_bits_correctness(alt_quantized_bits, kwargs)

    print("\nLinear accuracy tests passed!\n")


def _check_quantized_bits_correctness(alt_quantized_bits, kwargs):
    """Check that the alt_quantized_bits and qkeras.quantized_bits
    return the same result for all test values"""

    bits = kwargs.get("bits", 8)
    integer = kwargs.get("integer", 0)
    keep_negative = kwargs.get("keep_negative", True)
    alpha = kwargs.get("alpha", None)
    symmetric = kwargs.get("symmetric", False)
    # decidedly raises an error
    if bits < integer + keep_negative:
        return
    # Not implemented in qkeras
    if alpha in ("auto", "auto_po2") and (not symmetric or not keep_negative):
        return
    # bug in qkeras
    if bits - keep_negative == 0 and alpha in ("auto", "auto_po2"):
        return

    baseline = quantized_bits(**kwargs)
    alt = alt_quantized_bits(**kwargs)

    for x in TEST_X_VALUES:
        _check_correctness(alt, baseline, x, kwargs)


def _check_correctness(alt_func, baseline_func, x, kwargs):
    """Check that the alt_func and baseline_func return the same result for x"""

    with tf.device("GPU:0"):
        baseline_res = baseline_func(x).numpy()
        alt_res = alt_func(x).numpy()
        baseline_scale = np.array(baseline_func.scale)
        alt_scale = np.array(alt_func.scale)
    err_msg = (
        f"Failed for {kwargs} with {x = }. \n"
        f"{baseline_res = }, {alt_res = }. \n"
        f"{baseline_scale = }, {alt_scale = }"
    )
    if not np.allclose(baseline_res, alt_res):
        assert False, err_msg
        # import pdb; pdb.set_trace()
    if not np.allclose(baseline_scale, alt_scale) and K.max(x) > 0:
        assert False, err_msg
        # import pdb; pdb.set_trace()


get_time_info(__file__)


#############################################
# TERRIBLE HORRIBLE COPY PASTE FROM QKERAS
#############################################

import six
import re
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
from six.moves import range
from tensorflow.keras import initializers
from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.python.framework import smart_cond as tf_utils
from qkeras.quantizers import safe_eval
from pprint import pprint


class BaseQuantizer(tf.Module):
    """Base quantizer

    Defines behavior all quantizers should follow.
    """

    def __init__(self):
        self.built = False

    def build(self, var_name=None, use_variables=False):
        if use_variables:
            if hasattr(self, "qnoise_factor"):
                self.qnoise_factor = tf.Variable(
                    lambda: tf.constant(self.qnoise_factor, dtype=tf.float32),
                    name=_create_variable_name("qnoise_factor", var_name=var_name),
                    dtype=tf.float32,
                    trainable=False,
                )
            if hasattr(self, "integer"):
                self.integer = tf.Variable(
                    lambda: tf.constant(self.integer, dtype=tf.int32),
                    name=_create_variable_name("integer", var_name=var_name),
                    dtype=tf.int32,
                    trainable=False,
                )
        self.built = True

    def _set_trainable_parameter(self):
        pass

    def update_qnoise_factor(self, qnoise_factor):
        """Update qnoise_factor."""
        if isinstance(self.qnoise_factor, tf.Variable):
            # self.qnoise_factor is a tf.Variable.
            # This is to update self.qnoise_factor during training.
            self.qnoise_factor.assign(qnoise_factor)
        else:
            if isinstance(qnoise_factor, tf.Variable):
                # self.qnoise_factor is a numpy variable, and qnoise_factor is a
                # tf.Variable.
                self.qnoise_factor = qnoise_factor.eval()
            else:
                # self.qnoise_factor and qnoise_factor are numpy variables.
                # This is to set self.qnoise_factor before building
                # (creating tf.Variable) it.
                self.qnoise_factor = qnoise_factor

    # Override not to expose the quantizer variables.
    @property
    def variables(self):
        return ()

    # Override not to expose the quantizer variables.
    @property
    def trainable_variables(self):
        return ()

    # Override not to expose the quantizer variables.
    @property
    def non_trainable_variables(self):
        return ()


class quantized_bits(BaseQuantizer):  # pylint: disable=invalid-name
    """Quantizes the number to a number of bits.

    In general, we want to use a quantization function like:

    a = (pow(2,bits) - 1 - 0) / (max(x) - min(x))
    b = -min(x) * a

    in the equation:

    xq = a x + b

    This requires multiplication, which is undesirable. So, we
    enforce weights to be between -1 and 1 (max(x) = 1 and min(x) = -1),
    and separating the sign from the rest of the number as we make this function
    symmetric, thus resulting in the following approximation.

    1) max(x) = +1, min(x) = -1
    2) max(x) = -min(x)

    a = pow(2,bits-1)
    b = 0

    Finally, just remember that to represent the number with sign, the
    largest representation is -pow(2,bits) to pow(2, bits-1)

    Symmetric and keep_negative allow us to generate numbers that are symmetric
    (same number of negative and positive representations), and numbers that
    are positive.

    Note:
      the behavior of quantized_bits is different than Catapult HLS ac_fixed
      or Vivado HLS ap_fixed. For ac_fixed<word_length, integer_lenth, signed>,
      when signed = true, it is equavlent to
      quantized_bits(word_length, integer_length-1, keep_negative=True)

    Attributes:
      bits: number of bits to perform quantization.
      integer: number of bits to the left of the decimal point.
      symmetric: if true, we will have the same number of values for positive
        and negative numbers.
      alpha: a tensor or None, the scaling factor per channel.
        If None, the scaling factor is 1 for all channels.
      keep_negative: if true, we do not clip negative numbers.
      use_stochastic_rounding: if true, we perform stochastic rounding.
      scale_axis: which axis to calculate scale from
      qnoise_factor: float. a scalar from 0 to 1 that represents the level of
        quantization noise to add. This controls the amount of the quantization
        noise to add to the outputs by changing the weighted sum of
        (1 - qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
      var_name: String or None. A variable name shared between the tf.Variables
        created in the build function. If None, it is generated automatically.
      use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
          not.
      use_variables: Bool. Whether to make the quantizer variables to be dynamic
        tf.Variables or not.

    Returns:
      Function that computes fixed-point quantization with bits.
    """

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
        super(quantized_bits, self).__init__()
        self.bits = bits
        self.integer = integer
        self.symmetric = symmetric
        self.keep_negative = keep_negative
        self.alpha = alpha
        self.use_stochastic_rounding = use_stochastic_rounding
        # "auto*" |-> symmetric
        if isinstance(self.alpha, six.string_types):
            self.symmetric = True
        self.scale = None
        self.scale_axis = scale_axis
        self.qnoise_factor = qnoise_factor
        self.use_ste = use_ste
        self.var_name = var_name
        self.use_variables = use_variables

    def __str__(self):
        # Convert Tensors to printable strings by converting to a numpy array and
        # then using regex to remove brackets when there is only one integer bit
        integer_bits = re.sub(
            r"\[(\d)\]",
            r"\g<1>",
            str(
                self.integer.numpy()
                if isinstance(self.integer, tf.Variable)
                else self.integer
            ),
        )

        flags = [str(self.bits), integer_bits, str(int(self.symmetric))]
        if not self.keep_negative:
            flags.append("keep_negative=False")
        if self.alpha:
            alpha = str(self.alpha)
            if isinstance(self.alpha, six.string_types):
                alpha = "'" + alpha + "'"
            flags.append("alpha=" + alpha)
        if self.use_stochastic_rounding:
            flags.append(
                "use_stochastic_rounding=" + str(int(self.use_stochastic_rounding))
            )
        return "quantized_bits(" + ",".join(flags) + ")"

    def __call__(self, x):
        """Computes fixedpoint quantization of x."""
        if not self.built:
            self.build(var_name=self.var_name, use_variables=self.use_variables)

        x = K.cast_to_floatx(x)

        # quantized_bits with "1" bit becomes a binary implementation.
        unsigned_bits = self.bits - self.keep_negative
        m = K.cast_to_floatx(pow(2, unsigned_bits))
        m_i = K.cast_to_floatx(K.pow(2, self.integer))

        if self.alpha is None:
            scale = 1.0
        elif isinstance(self.alpha, six.string_types):
            # We only deal with the symmetric case right now.
            assert self.symmetric, "Only symmetric quantizers are implemented"
            len_axis = len(x.shape)
            if len_axis != 1:
                axis = _get_scaling_axis(self.scale_axis, len_axis)
            else:
                axis = [0]

            x = x / m_i

            # Using 2's complement, we can represent 2**(bits-1)-1 positive values
            # If we wish to maintain symmetry, we can double 2**(bits-1)-1 to get
            # the total number of possible values we can represent.
            # If symmetry is not enforced, then we can represent (2**bits)-1 values
            # using 2's complement.
            levels = (
                (2 ** (self.bits - 1) - 1) * 2
                if self.symmetric
                else (2**self.bits) - 1
            )

            scale = (K.max(abs(x), axis=axis, keepdims=True) * 2) / levels

            # If alpha is "auto_po2", then get the "best" po2 scale
            if "po2" in self.alpha:
                scale = K.pow(
                    2.0, tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0))
                )

                for _ in range(5):
                    v = tf.floor(tf.abs(x) / scale + 0.5)
                    mask = v < levels / 2
                    z = tf.sign(x) * tf.where(mask, v, tf.ones_like(v) * levels / 2)
                    scale = _get_scale(
                        alpha="auto_po2", x=x, q=z, scale_axis=self.scale_axis
                    )

            # If alpha is "auto", then get the "best" floating point scale
            elif self.alpha == "auto":
                v = tf.floor(tf.abs(x) / scale + 0.5)
                mask = v < levels / 2
                z = tf.sign(x) * tf.where(mask, v, tf.ones_like(v) * levels / 2)
            else:
                raise ValueError(f"Invalid alpha '{self.alpha}'")
            # z is an integer number, so we must make the scale * m and z / m
            scale = scale * m

            # we will not use "z" right now because of stochastic_rounding
            # this is still under test.

            # if "new" in self.alpha:
            #  z = z / m
            #  self.scale = scale
            #  return x + tf.stop_gradient(-x + scale * z)
            x = m_i * x
            xq = m_i * z / m
            self.scale = scale
            xq = scale * xq

            if self.use_ste:
                return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
            else:
                return (1 - self.qnoise_factor) * x + tf.stop_gradient(
                    self.qnoise_factor * xq
                )

        else:
            scale = self.alpha

        # quantized_bits with "1" bit becomes a binary implementation.
        if unsigned_bits > 0:
            p = x * m / m_i
            xq = (
                m_i
                * tf.keras.backend.clip(
                    _round_through(p, self.use_stochastic_rounding, precision=1.0),
                    self.keep_negative * (-m + self.symmetric),
                    m - 1,
                )
                / m
            )
        else:
            xq = tf.sign(x)
            xq += 1.0 - tf.abs(xq)
            if not self.keep_negative:
                xq = (xq + 1.0) / 2.0

        self.scale = scale
        xq = scale * xq

        if self.use_ste:
            return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
        else:
            return (1 - self.qnoise_factor) * x + tf.stop_gradient(
                self.qnoise_factor * xq
            )

    def _set_trainable_parameter(self):
        if self.alpha is None:
            self.alpha = "auto_po2"
            self.symmetric = True

    def max(self):
        """Get maximum value that quantized_bits class can represent."""
        unsigned_bits = self.bits - self.keep_negative
        if unsigned_bits > 0:
            return max(
                1.0,
                np.array(
                    K.pow(2.0, K.cast(self.integer, dtype="float32")), dtype="float32"
                ),
            )
        else:
            return 1.0

    def min(self):
        """Get minimum value that quantized_bits class can represent."""
        if not self.keep_negative:
            return 0.0
        unsigned_bits = self.bits - self.keep_negative
        if unsigned_bits > 0:
            return -max(
                1.0,
                np.array(
                    K.pow(2, K.cast(self.integer, dtype="float32")), dtype="float32"
                ),
            )
        else:
            return -1.0

    def range(self):
        """Returns a list of all values that quantized_bits can represent
        ordered by their binary representation ascending."""
        assert self.symmetric == 0
        assert self.keep_negative
        assert self.alpha is None or self.alpha == 1.0

        x = np.asarray(range(2**self.bits), dtype=np.float32)
        p_and_n = np.where(
            x >= 2 ** (self.bits - 1),
            (x - 2 ** (self.bits - 1)) - 2 ** (self.bits - 1),
            x,
        )
        return p_and_n * np.array(
            K.pow(2.0, -self.bits + K.cast(self.integer, dtype="float32") + 1),
            dtype="float32",
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "bits": self.bits,
            "integer": self.integer.numpy()
            if isinstance(self.integer, tf.Variable)
            else self.integer,
            "symmetric": self.symmetric,
            "alpha": self.alpha,
            "keep_negative": self.keep_negative,
            "use_stochastic_rounding": self.use_stochastic_rounding,
            "qnoise_factor": self.qnoise_factor.numpy()
            if isinstance(self.qnoise_factor, tf.Variable)
            else self.qnoise_factor,
        }
        return config


#
# Library of auxiliary functions
#


def _get_scaling_axis(scale_axis, len_axis):
    """Get the axis to perform auto scaling with."""

    if scale_axis is not None:
        axis = list(range(scale_axis))
        axis += list(range(scale_axis + 1, len_axis))
    else:
        if K.image_data_format() == "channels_last":
            axis = list(range(len_axis - 1))
        else:
            axis = list(range(1, len_axis))
    return axis


def _get_scale(alpha, x, q, scale_axis=None, per_channel_scale=True):
    """Gets scaling factor for scaling the tensor per channel.
    It uses the least squares method to find the scaling factor.

    (https://en.wikipedia.org/wiki/Linear_least_squares)

    Arguments:
      alpha: A float or string. When it is string, it should be either "auto" or
        "auto_po2", and scale = sum(x * q, axis=all but last) / sum(q * q,
        axis=all but last)
       x: A tensor object. Its elements are in float.
       q: A tensor object. Its elements are in quantized format of x.
       scale_axis: which axis to calculate scale from
       per_channel_scale: A bool. Whether to perform per-channel scaling or not.

    Returns:
      A scaling factor tensor or scalar for scaling tensor per channel.
    """

    if isinstance(alpha, six.string_types) and "auto" in alpha:
        assert alpha in ["auto", "auto_po2"]
        # in different tensorflow version (e.g., 2.4)
        # x.shape is a tuple which doesn't have as_list() method
        try:
            x_shape = x.shape.as_list()
        except AttributeError:
            x_shape = list(x.shape)

        len_axis = len(x_shape)
        if not per_channel_scale:
            qx = K.mean(x * q, keepdims=True)
            qq = K.mean(q * q, keepdims=True)
        else:
            if len_axis > 1:
                axis = _get_scaling_axis(scale_axis, len_axis)
                qx = K.mean(tf.math.multiply(x, q), axis=axis, keepdims=True)
                qq = K.mean(tf.math.multiply(q, q), axis=axis, keepdims=True)
            else:
                # No summing (averaging) along the channel axis to get per-channel
                # scales.
                qx = x * q
                qq = q * q

        scale = qx / (qq + K.epsilon())
        if alpha == "auto_po2":
            scale = K.pow(2.0, tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0)))
    elif alpha is None:
        scale = 1.0
    elif isinstance(alpha, np.ndarray):
        scale = alpha
    else:
        scale = float(alpha)
    return scale


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
    if use_stochastic_rounding:
        output = tf_utils.smart_cond(
            K.learning_phase(),
            lambda: x + tf.stop_gradient(-x + stochastic_round(x, precision)),
            lambda: x + tf.stop_gradient(-x + tf.round(x)),
        )
    else:
        output = x + tf.stop_gradient(-x + tf.round(x))
    return output


def _create_variable_name(attr_name, var_name=None):
    """Creates variable name.
    Arguments:
      attr_name: string. attribute name
      var_name: string. variable name

    Returns:
      string. variable name
    """

    if var_name:
        return var_name + "/" + attr_name

    # This naming scheme is to solve a problem of a layer having more than
    # one quantizer can have multiple qnoise_factor variables with the same
    # name of "qnoise_factor".
    return attr_name + "_" + str(K.get_uid(attr_name))


def stochastic_round(x, precision=0.5):
    """Performs stochastic rounding to the first decimal point."""
    scale = 1.0 / precision
    scale_x = x * scale
    fraction = scale_x - tf.floor(scale_x)

    result = tf.where(
        fraction < tf.random.uniform(tf.shape(x)),
        tf.math.floor(scale_x),
        tf.math.ceil(scale_x),
    )
    return result / scale


#
# Activation functions for quantized networks.
#
# Please note some of these functions can be used as well
# as quantizer functions for weights of dense and convolutional
# layers.
#
