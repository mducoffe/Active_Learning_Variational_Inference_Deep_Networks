# dropout operator on ConvolutionActivation, ConvolutionalLayer, MLP
# so to train a network without ComputationGraph using dropout
import logging

import numpy
from six import add_metaclass
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from toolz import interleave
from picklable_itertools.extras import equizip

from blocks.config import config
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS
#from blocks.utils import pack, shared_floatx_nans, named_copy
from blocks.bricks import (Initializable, Feedforward,
                            Sequence, Linear, Identity)
from blocks.bricks.conv import (ConvolutionalActivation, ConvolutionalLayer,
                                ConvolutionalSequence, Flattener)

class Dropout(Initializable):

    @lazy(allocation=['drop_prob'])
    def __init__(self, drop_prob, **kwargs):
        assert drop_prob>=0 and drop_prob<1
        self.drop_prob = drop_prob
        super(Dropout, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply an amount of dropout on the output
        """

        rng = MRG_RandomStreams(config.default_seed)
        return input_*rng.binomial(input_.shape, p=1 - self.drop_prob, dtype=theano.config.floatX)#/(1 - self.drop_prob)
        """
        if input_.ndim == 2:
            return input_*rng.binomial(input_.shape, p=1 - self.drop_prob, dtype=theano.config.floatX)/(1 - self.drop_prob)
        elif input_.ndim ==4:
            var = rng.binomial(input_.shape[:2], p=1-self.drop_prob, dtype=theano.config.floatX)/(1 - self.drop_prob)
            return input_*var.dimshuffle((0, 1, 'x', 'x'))
        else :
            raise Exception('unknow dimension of input_ %d', input_.ndim)
        """

class MLP_dropout(Sequence, Initializable, Feedforward):
    """A simple multi-layer perceptron.
    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.
    Notes
    -----
    See :class:`Initializable` for initialization parameters.
    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()
    """
    @lazy(allocation=['dims', 'drop_probs'])
    def __init__(self, activations, dims, drop_probs, **kwargs):
        self.activations = activations
        self.drop_probs = drop_probs
        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        # Interleave the transformations and activations
        application_methods = []
        i = 0
        for entity in interleave([self.linear_transformations, activations]):
            if entity is None:
                continue
            if isinstance(entity, Brick):
                application_methods.append(entity.apply)
            else:
                application_methods.append(entity)
                application_methods.append(Dropout(drop_probs[i]).apply)
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP_dropout, self).__init__(application_methods, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias


class ConvolutionalSequence_dropout(Sequence, Initializable, Feedforward):
    """A sequence of convolutional operations.
    Parameters
    ----------
    layers : list
        List of convolutional bricks (i.e. :class:`ConvolutionalActivation`
        or :class:`ConvolutionalLayer`)
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer.
    batch_size : int, optional
        Number of images in batch. If given, will be passed to
        theano's convolution operator resulting in possibly faster
        execution.
    image_size : tuple, optional
        Width and height of the input (image/featuremap). If given,
        will be passed to theano's convolution operator resulting in
        possibly faster execution.
    Notes
    -----
    The passed convolutional operators should be 'lazy' constructed, that
    is, without specifying the batch_size, num_channels and image_size. The
    main feature of :class:`ConvolutionalSequence` is that it will set the
    input dimensions of a layer to the output dimensions of the previous
    layer by the :meth:`~.Brick.push_allocation_config` method.
    """
    @lazy(allocation=['num_channels', 'drop_probs'])
    def __init__(self, layers, num_channels, drop_probs, batch_size=None, image_size=None,
                 border_mode='valid', tied_biases=False, **kwargs):
        self.layers = layers
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.tied_biases = tied_biases
        self.drop_probs = drop_probs
        application_methods_ = [brick.apply for brick in layers]
        application_methods = []
        for app, i in zip(application_methods_, xrange(len(application_methods_))):
            application_methods.append(app)
            if i < len(application_methods_):
                application_methods.append(Dropout(drop_probs).apply)
        super(ConvolutionalSequence_dropout, self).__init__(
            application_methods=application_methods, **kwargs)

    def get_dim(self, name):
        if name == 'input_':
            return ((self.num_channels,) + self.image_size)
        if name == 'output':
            return self.layers[-1].get_dim(name)
        return super(ConvolutionalSequence, self).get_dim(name)

    def _push_allocation_config(self):
        num_channels = self.num_channels
        image_size = self.image_size
        for layer in self.layers:
            for attr in ['border_mode', 'tied_biases']:
                setattr(layer, attr, getattr(self, attr))
            layer.image_size = image_size
            layer.num_channels = num_channels
            layer.batch_size = self.batch_size

            # Push input dimensions to children
            layer._push_allocation_config()

            # Retrieve output dimensions
            # and set it for next layer
            if layer.image_size is not None:
                output_shape = layer.get_dim('output')
                image_size = output_shape[1:]
            num_channels = layer.num_filters
