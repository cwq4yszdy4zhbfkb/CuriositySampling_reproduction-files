import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from deel.lip.layers import SpectralDense, FrobeniusDense
from deel.lip.initializers import BjorckInitializer
from deel.lip.activations import FullSort, PReLUlip, MaxMin, GroupSort2
from curiositysampling.utils import Psnake

# a custom initializer, that returns uniform distributed
# weights with weights shifted from zero
# to prevent small weights
class NonZeroHeUniform(tf.keras.initializers.HeUniform):
    """ """

    def __init__(self, seed=None, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, shape, dtype=None):
        weights = super().__call__(shape, dtype)
        negative_mask = tf.cast(weights < 0, dtype)
        positive_mask = tf.cast(weights >= 0, dtype)
        # for all negative weights add negative epsilon
        # and for all positive, add positive epsilon
        weights += -self.epsilon * negative_mask + self.epsilon * positive_mask
        return weights


class Orthonormal(tf.keras.constraints.Constraint):
    """approximate Orthonormal weight constraint.
    Constrains the weights incident to each hidden unit
    to be approximately orthonormal

    # Arguments
        beta: the strength of the constraint

    # References
        https://arxiv.org/pdf/1710.04087.pdf
    """

    def __init__(self, beta=0.01):
        self.beta = beta

    def __call__(self, w):
        eye = tf.linalg.matmul(w, w, transpose_b=True)
        return (1 + self.beta) * w - self.beta * tf.linalg.matmul(eye, w)

    def get_config(self):
        return {"beta": self.beta}


# Thanks for the code
# https://gist.github.com/aeftimia/a5249168c84bc541ace2fc4e1d22a13e
class Orthogonal(tf.keras.constraints.Constraint):
    """Orthogonal weight constraint.
    Constrains the weights incident to each hidden unit
    to be orthogonal when there are more inputs than hidden units.
    When there are more hidden units than there are inputs,
    the rows of the layer's weight matrix are constrainted
    to be orthogonal.
    # Arguments
        axis: Axis or axes along which to calculate weight norms.
            `None` to use all but the last (output) axis.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
        orthonormal: If `True`, the weight matrix is further
            constrained to be orthonormal along the appropriate axis.
    """

    def __init__(self, axis=None, orthonormal=False):
        if axis is not None:
            self.axis = axis
        else:
            self.axis = None
        self.orthonormal = orthonormal

    def __call__(self, w):
        # Python block for permutating axis
        w_ndim_minus_1 = len(w.shape) - 1
        if self.axis is None:
            self.axis = tf.range(w_ndim_minus_1)
        elif isinstance(self.axis, int):
            self.axis = [self.axis]

        axis_shape = []
        for a in self.axis:
            w_shape = w.shape[a]
            axis_shape.append(w_shape)

        perm = []
        for i in range(w_ndim_minus_1):
            if i not in self.axis:
                perm.append(i)
        perm.extend(self.axis)
        perm.append(w_ndim_minus_1)

        w = tf.transpose(w, perm=perm)
        shape = w.shape
        new_shape = [-1] + axis_shape + [shape[-1]]
        w = tf.reshape(w, new_shape)
        w = tf.map_fn(self.orthogonalize, w)
        w = tf.reshape(w, shape)
        w = tf.transpose(w, perm=tf.argsort(perm))
        return w

    def orthogonalize(self, w):
        shape = w.shape
        output_shape = tf.convert_to_tensor(shape[-1], dtype=tf.int32)
        input_shape = tf.math.reduce_prod(shape[:-1])
        final_shape = tf.math.maximum(input_shape, output_shape)
        w_matrix = tf.reshape(w, (output_shape, input_shape))
        zero_int = tf.constant(0, dtype=tf.int32)
        paddings = tf.convert_to_tensor(
            [
                [zero_int, final_shape - output_shape],
                [zero_int, final_shape - input_shape],
            ]
        )
        w_matrix = tf.pad(w_matrix, paddings)
        upper_triangular = tf.linalg.band_part(w_matrix, 1, -1)
        antisymmetric = upper_triangular - tf.transpose(upper_triangular)
        rotation = tf.linalg.expm(antisymmetric)
        w_final = tf.slice(
            rotation,
            [
                0,
            ]
            * 2,
            [output_shape, input_shape],
        )
        if not self.orthonormal:
            w_final = tf.cond(
                tf.math.greater_equal(input_shape, output_shape),
                lambda: tf.linalg.matmul(
                    w_final,
                    tf.linalg.band_part(
                        tf.slice(w_matrix, [0, 0], [input_shape, input_shape]), 0, 0
                    ),
                ),
                lambda: tf.linalg.matmul(
                    tf.linalg.band_part(
                        tf.slice(w_matrix, [0, 0], [output_shape, output_shape]), 0, 0
                    ),
                    w_final,
                ),
            )
        return tf.reshape(w_final, w.shape)

    def get_config(self):
        return {"axis": self.axis, "orthonormal": self.orthonormal}


def orthogonality(w):
    """
    Penalty for deviation from orthogonality:
    Orthogonalize column vectors
    ||dot(x, x.T) - I||_1
    """
    wTw = tf.matmul(w, w, transpose_b=True)
    return tf.norm(wTw - tf.eye(*wTw.shape), ord=1)


class RND(tf.keras.Model):
    def __init__(
        self,
        dense_units=[64],
        dense_activ="relu",
        dense_layernorm=False,
        dense_batchnorm=False,
        dense_out=1,
        dense_out_activ="sigmoid",
        layernorm_out=False,
        target=False,
        initializer="he_uniform",
        spectral=False,
        curiosity_activ=None,
        orthonormal=False,
        l1_reg=0.0001,
        l2_reg=0.0001,
        unit_constraint=False,
    ):

        super().__init__(self)
        if not spectral:
            self.initializer = initializer
        else:
            self.initializer = "orthogonal"
        self.dense_layers = []
        self.layernorm_layers = []
        self.activation_layers = []
        # dense layers
        for units_layer in dense_units:
            if not spectral:
                if curiosity_activ is not None:
                    dense_activ = curiosity_activ
                if dense_activ == "prelu":
                    dense_activ_lay = tf.keras.layers.PReLU()
                elif dense_activ == "lerelu":
                    dense_activ_lay = tf.keras.layers.LeakyReLU(alpha=0.01)
                elif dense_activ == "snake":
                    dense_activ_lay = tfa.activations.snake
                elif dense_activ == "psnake":
                    dense_activ_lay = Psnake()
                elif dense_activ == "gelu":
                    dense_activ_lay = tfa.activations.gelu
                elif dense_activ == "lisht":
                    dense_activ_lay = tfa.activations.lisht
                elif dense_activ == "mish":
                    dense_activ_lay = tfa.activations.mish
                else:
                    dense_activ_lay = tf.keras.layers.Activation(dense_activ)
                if curiosity_activ is not None:
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            units_layer,
                            kernel_initializer=self.initializer,
                        )
                    )
                else:
                    if unit_constraint:
                        constraint = tf.keras.constraints.UnitNorm()
                    else:
                        constraint = None
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            units_layer,
                            kernel_initializer=self.initializer,
                            kernel_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            kernel_constraint=constraint,
                        )
                    )
                self.activation_layers.append(dense_activ_lay)
            else:
                if dense_activ == "prelulip":
                    activation = PReLUlip()
                elif dense_activ == "fullsort":
                    activation = FullSort()
                elif dense_activ == "groupsort2":
                    activation = GroupSort2()
                elif dense_activ == "maxmin":
                    activation = MaxMin()
                else:
                    tf.print(
                        "Warning, non-Lipschitz function is used, while using Lipschitz"
                    )
                    activation = tf.keras.layers.Activation(dense_activ)
                self.dense_layers.append(
                    SpectralDense(
                        units_layer,
                        kernel_initializer=BjorckInitializer(15, 50),
                        k_coef_lip=1.0,
                    )
                )

                dense_activ_lay = activation
                self.activation_layers.append(dense_activ_lay)

            if dense_layernorm and not target and not curiosity_activ:
                self.layernorm_layers.append(tf.keras.layers.LayerNormalization())
            elif dense_batchnorm and not target and not curiosity_activ:
                self.layernorm_layers.append(tf.keras.layers.BatchNormalization())
            else:
                self.layernorm_layers.append(tf.keras.layers.Activation("linear"))

        if not spectral:
            if curiosity_activ is not None:
                self.dense_out = tf.keras.layers.Dense(
                    dense_out,
                    activation=dense_out_activ,
                    kernel_initializer=self.initializer,
                    use_bias=True,
                )
            else:
                if orthonormal:
                    self.dense_out = tf.keras.layers.Dense(
                        dense_out,
                        activation=dense_out_activ,
                        kernel_initializer="orthogonal",
                        kernel_constraint=Orthonormal(beta=0.1),
                        use_bias=True,
                    )
                else:

                    if unit_constraint:
                        constraint = tf.keras.constraints.UnitNorm()
                    else:
                        constraint = None

                    self.dense_out = tf.keras.layers.Dense(
                        dense_out,
                        activation=dense_out_activ,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg, l2=l2_reg
                        ),
                        use_bias=True,
                        kernel_constraint=constraint,
                    )

        elif dense_out > 1:
            self.dense_out = SpectralDense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer=BjorckInitializer(15, 50),
                use_bias=True,
                k_coef_lip=1.0,
            )
        else:
            self.dense_out = FrobeniusDense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer=BjorckInitializer(15, 50),
                use_bias=True,
                k_coef_lip=1.0,
            )
        if layernorm_out and not target:
            self.layernorm_out = tf.keras.layers.LayerNormalization()
        else:
            self.layernorm_out = tf.keras.layers.Activation("linear")
        assert len(self.dense_layers) != 0
        assert len(self.layernorm_layers) != 0
        assert len(self.activation_layers) != 0
        assert len(self.activation_layers) == len(self.dense_layers)
        assert len(self.layernorm_layers) == len(self.dense_layers)

    def call(self, inputs, training=False):
        x = inputs
        for dense_layer, layernorm_layer, activation_layer in zip(
            self.dense_layers, self.layernorm_layers, self.activation_layers
        ):
            x = dense_layer(x, training=training)
            x = layernorm_layer(x, training=training)
            x = activation_layer(x)

        x = self.dense_out(x, training=training)
        x = self.layernorm_out(x, training=training)

        return x


class RND_decoder(tf.keras.Model):
    def __init__(
        self,
        dense_units=[64],
        dense_activ="relu",
        dense_layernorm=False,
        dense_batchnorm=True,
        dense_out=1,
        dense_out_activ="sigmoid",
        target=False,
        initializer="he_uniform",
        spectral=False,
    ):

        super().__init__(self)
        # init initializer
        if target and not spectral:
            self.initializer = NonZeroHeUniform(epsilon=0.1)
        elif not spectral:
            self.initializer = initializer
        else:
            self.initializer = "orthogonal"
        self.dense_layers = []
        self.layernorm_layers = []
        self.activation_layers = []
        # dense layers
        for units_layer in dense_units:
            if not spectral:
                if dense_activ == "prelu":
                    dense_activ_lay = tf.keras.layers.PReLU()
                elif dense_activ == "lerelu":
                    dense_activ_lay = tf.keras.layers.LeakyReLU(alpha=0.01)
                elif dense_activ == "snake":
                    dense_activ_lay = tfa.activations.snake
                elif dense_activ == "psnake":
                    dense_activ_lay = Psnake()
                elif dense_activ == "gelu":
                    dense_activ_lay = tfa.activations.gelu
                elif dense_activ == "lisht":
                    dense_activ_lay = tfa.activations.lisht
                elif dense_activ == "mish":
                    dense_activ_lay = tfa.activations.mish
                else:
                    dense_activ_lay = tf.keras.layers.Activation(dense_activ)

                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        units_layer,
                        kernel_initializer=self.initializer,
                    )
                )
                self.activation_layers.append(dense_activ_lay)
            else:
                if dense_activ == "prelulip":
                    activation = PReLUlip()
                elif dense_activ == "fullsort":
                    activation = FullSort()
                elif dense_activ == "groupsort2":
                    activation = GroupSort2()
                elif dense_activ == "maxmin":
                    activation = MaxMin()
                else:
                    tf.print(
                        "Warning, non-Lipschitz function is used, while using Lipschitz"
                    )
                    activation = tf.keras.layers.Activation(dense_activ)
                self.dense_layers.append(
                    SpectralDense(
                        units_layer,
                        kernel_initializer=BjorckInitializer(15, 50),
                    )
                )
                dense_activ_lay = activation
                self.activation_layers.append(dense_activ_lay)

            if dense_layernorm and not target:
                self.layernorm_layers.append(tf.keras.layers.LayerNormalization())
            elif dense_batchnorm and not target:
                self.layernorm_layers.append(tf.keras.layers.BatchNormalization())

            else:
                self.layernorm_layers.append(tf.keras.layers.Activation("linear"))
        if not spectral:
            self.dense_out = tf.keras.layers.Dense(
                dense_out,
                activation="tanh",
                kernel_initializer=self.initializer,
            )
        elif dense_out > 1:
            self.dense_out = SpectralDense(
                dense_out,
                activation="tanh",
                kernel_initializer=BjorckInitializer(15, 50),
            )
        else:
            self.dense_out = FrobeniusDense(
                dense_out,
                activation="tanh",
                kernel_initializer=BjorckInitializer(15, 50),
            )
        assert len(self.dense_layers) != 0
        assert len(self.layernorm_layers) != 0
        assert len(self.activation_layers) != 0
        assert len(self.activation_layers) == len(self.dense_layers)
        assert len(self.layernorm_layers) == len(self.dense_layers)

    def call(self, inputs, training=False):
        x = inputs
        for dense_layer, layernorm_layer, activation_layer in zip(
            self.dense_layers, self.layernorm_layers, self.activation_layers
        ):
            x = dense_layer(x, training=training)
            x = layernorm_layer(x, training=training)
            x = activation_layer(x)

        x = self.dense_out(x, training=training)

        return x
