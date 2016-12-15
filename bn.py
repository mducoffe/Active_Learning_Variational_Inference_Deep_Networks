#### batch_normalization ################################
#### new version cause blocks is always evolving =( #####
# @date 18/02/2016 ######################################
# @author Melanie Ducoffe ###############################
#########################################################

import numpy
from blocks import bricks
from blocks.bricks import Sequence
from blocks.bricks import conv
from blocks.bricks.base import application, lazy
from blocks.utils import shared_floatx_nans
from theano import tensor
from theano.tensor.nnet.conv import ConvOp
from theano.sandbox.cuda.dnn import dnn_conv
from blocks.bricks import Initializable, Feedforward, Sequence

######
import logging

import numpy
from six import add_metaclass
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from toolz import interleave
from picklable_itertools.extras import equizip

from blocks.config import config
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER
from blocks.utils import pack, shared_floatx_nans
###########

class Linear(bricks.Linear):
    """Linear brick with batch normalization applied.

    Parameters
    ----------
    init_log_gamma : float, optional
        Initial value for gamma in log-space. Defaults to 0.0.

    """
    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, init_log_gamma=0.0, **kwargs):
        super(Linear, self).__init__(
            input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.init_log_gamma = init_log_gamma

    def _allocate(self):
        super(Linear, self)._allocate()
        log_gamma = shared_floatx_nans((self.output_dim,), name='log_gamma')
        beta = shared_floatx_nans((self.output_dim,), name='beta')
        self.parameters.append(log_gamma)
        self.parameters.append(beta)
	add_role(log_gamma, WEIGHT)
	add_role(beta, BIAS)

    def _initialize(self):
        if self.use_bias:
            W, b, log_gamma, beta = self.parameters
            self.biases_init.initialize(b, self.rng)
        else:
            W, log_gamma, beta = self.parameters
        self.weights_init.initialize(W, self.rng)
        log_gamma.set_value(
            self.init_log_gamma * numpy.ones_like(log_gamma.get_value()))
	beta.set_value(numpy.zeros_like(beta.get_value()))

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.use_bias:
            W, b, log_gamma, beta = self.parameters
        else:
            W, log_gamma, beta = self.parameters
        output = tensor.dot(input_, W)
        if self.use_bias:
            output += b

        gamma = tensor.exp(log_gamma).dimshuffle('x', 0)
        beta = beta.dimshuffle('x', 0)
        mean = output.mean(axis=0, keepdims=True)
        var = output.var(axis=0, keepdims=True)
        output = gamma * (output - mean) / tensor.sqrt(var + 1e-5) + beta

        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.output_dim
        super(Linear, self).get_dim(name)


class Convolutional(conv.Convolutional):
    """CuDNN-enabled convolutional brick with batch normalization applied.

    Parameters
    ----------
    init_log_gamma : float, optional
        Initial value for gamma in log-space. Defaults to 0.0.

    """

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
                 image_size=(None,None), step=(1, 1), border_mode='valid',
                 tied_biases=False, init_log_gamma=1.0, **kwargs):
        super(Convolutional, self).__init__(
            filter_size=filter_size, num_filters=num_filters,
            num_channels=num_channels, batch_size=batch_size,
            image_size=image_size, step=step, border_mode=border_mode,
            tied_biases=tied_biases, **kwargs)
        self.init_log_gamma = init_log_gamma

    def _allocate(self):
        super(Convolutional, self)._allocate()
        log_gamma = shared_floatx_nans((self.num_filters,), name='log_gamma')
        beta = shared_floatx_nans((self.num_filters,), name='beta')
        self.parameters.append(log_gamma)
        self.parameters.append(beta)
	add_role(log_gamma, WEIGHT)
	add_role(beta, BIAS)

    def _initialize(self):
        if self.use_bias:
            W, b, log_gamma, beta = self.parameters
            self.biases_init.initialize(b, self.rng)
        else:
            W, log_gamma, beta = self.parameters
        self.weights_init.initialize(W, self.rng)
        log_gamma.set_value(
            self.init_log_gamma * numpy.ones_like(log_gamma.get_value()))
        beta.set_value(numpy.zeros_like(beta.get_value()))

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.use_bias:
            W, b, log_gamma, beta = self.parameters
        else:
            W, log_gamma, beta = self.parameters

        if self.image_size == (None, None):
            image_shape = None
        else:
            image_shape = (self.batch_size, self.num_channels)
            image_shape += self.image_size

        output = self.conv2d_impl(
            input_, W,
            image_shape=image_shape,
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=((self.num_filters, self.num_channels) +
                          self.filter_size))
        if self.use_bias:
            if self.tied_biases:
                output += b.dimshuffle('x', 0, 'x', 'x')
            else:
                output += b.dimshuffle('x', 0, 1, 2)

	gamma = log_gamma.dimshuffle('x', 0, 'x', 'x')
        beta = beta.dimshuffle('x', 0, 'x', 'x')
        mean = output.mean(axis=[0, 2, 3], keepdims=True)
        var = output.var(axis=[0, 2, 3], keepdims=True)
        #output = gamma * (output - mean) / tensor.sqrt(var + 1e-5) + beta
	#output = gamma*(output-mean)/ tensor.sqrt(var + 1e-5)
	output = gamma*(output - mean) / tensor.sqrt(var + 1e-5) + beta
        return output

class ConvolutionalActivation(conv._AllocationMixin, Sequence, Initializable):
    """Convolutional activation with batch normalization applied.

    Parameters
    ----------
    init_log_gamma : float, optional
        Initial value for gamma in log-space. Defaults to 0.0.

    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, activation, filter_size, num_filters, num_channels,
                 batch_size=None, image_size=None, step=(1, 1),
                 border_mode='valid', tied_biases=False, init_log_gamma=1.0, **kwargs):

        self.convolution = Convolutional(init_log_gamma=init_log_gamma)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
	self.tied_biases = tied_biases
	self.activation=activation
	self.init_log_gamma=init_log_gamma
	super(ConvolutionalActivation, self).__init__(
            application_methods=[self.convolution.apply, activation],
            **kwargs)

    def get_dim(self, name):
        # TODO The name of the activation output doesn't need to be `output`
        return self.convolution.get_dim(name)

    def _push_allocation_config(self):
        super(ConvolutionalActivation, self)._push_allocation_config()
        self.convolution.step = self.step
