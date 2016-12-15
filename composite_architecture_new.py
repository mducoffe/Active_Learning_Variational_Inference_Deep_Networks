import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import (Initializable, Feedforward, Sequence,
                            Rectifier, Tanh, Logistic, Identity, MLP, Linear,
                            Softmax)
from blocks.bricks.conv import (ConvolutionalSequence, Flattener, MaxPooling, Convolutional)
from blocks.bricks.base import lazy
from blocks.initialization import Constant, Uniform, IsotropicGaussian

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks import Softmax
from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
from blocks.roles import WEIGHT, BIAS
#from bn import ConvolutionalActivation


# the new version of blocks needs upgrading the code 

class CompositeSequence(Sequence, Initializable, Feedforward):
    
    @lazy(allocation=[])
    def __init__(self, conv_seq=None, mlp=None, **kwargs):
        self.conv_seq = conv_seq;
        self.mlp = mlp;
        self.flatten = Flattener()
        self.T = 1.
        
        application_methods = [self.conv_seq.apply, self.flatten.apply,
                                self.mlp.apply]
        super(CompositeSequence, self).__init__(
            application_methods=application_methods, **kwargs)

    def set_T(self, newT):
        #assert newT >=1.
        self.T = newT

    def confidence(self, input_):
	y_prev = self.apply(input_)
	labels= T.cast(T.argmax(Softmax().apply(y_prev), axis=1), 'uint8')
	return Softmax().categorical_cross_entropy(labels.flatten(), y_prev)
            
    def error(self, x, y):
        y_pred = Softmax().apply(self.apply(x))
        return MisclassificationRate().apply(y.flatten(), y_pred).mean()

    def predict(self, x):
        return T.argmax(self.probabilities(x), axis=1)

    def probabilities(self, x):
	return Softmax().apply(self.apply(x))

    def _push_allocation_config(self):
        self.conv_seq._push_allocation_config()
        self.mlp._push_allocation_config()
        self.flatten._push_allocation_config()
        
    def get_dim(self, name):
        if name == 'input_':
            return self.conv_seq.get_dim('input_')
        if name == 'output':
	    return self.mlp.output_dim
            #return self.mlp.get_dim('output')
        return super(CompositeSequence, self).get_dim(name)
        
    def get_Params(self):

        # warning : sometimes with the gpu mode the list
        # of parameters can be from the output to the input layer
        # we need in this case to inverse the list
        
        # to check that we look at the list of parameters 
        # of the whole system
        # if the first weight is a matrix then inverse
        #return a dictionary of params
        x = T.tensor4()
        cost = self.apply(x).sum()
        cg = ComputationGraph(cost)
        W = VariableFilter(roles=[WEIGHT])(cg.variables)
        B = VariableFilter(roles=[BIAS])(cg.variables)
	if W[0].name[:5]=="layer":
		return W, B

	# find other parameters and retrieve them for the lists
	gamma=[]
	beta=[]
	index_gamma=[]; index_beta=[]
	for w, b, i in zip(W,B, range(len(W))):
		
		if w.name=="log_gamma":
			index_gamma.append(i); gamma.append(w)
		if b.name=="beta":
			index_beta.append(i); beta.append(b)

	for i in index_gamma[::-1] : W.pop(i)
	for i in index_beta[::-1] : B.pop(i)


	if len(W)==0:
		import pdb
		pdb.set_trace()
        if W[0].ndim == 2:
            W_ = []
            for  i in xrange(len(W)):
                W_.append(W[len(W) - 1 -i])
            W = W_
        if B[0].ndim == 1:
            B_ = []
            for  i in xrange(len(B)):
                B_.append(B[len(B)-1 -i])
            B = B_
        
	# if batch normalization has been introduced you need to reinject artificially
	# the gamma and beta parameters so to fit with the actual protocol of dropout
	if len(gamma) !=len(beta):
		raise Exception(" gamma and beta parameters should be balanced : (%d, %d)", len(gamma), len(beta))

	if len(gamma)!=0:
		if beta[0].shape.eval()!=gamma[0].shape.eval():
			beta.reverse()
		W_new=[]; B_new=[]
		for w, g in zip(W[:len(gamma)], gamma):
			W_new.append(w); W_new.append(g)
		W_new += W[len(gamma):]
		for b, b_ in zip(B[:len(gamma)], beta):
			B_new.append(b); B_new.append(b_)
		B_new += B[len(gamma):]
		W = W_new; B = B_new

        for w, b, index in zip(W, B, range(len(W))):
            w.name = "layer_"+str(index)+"_W"
            b.name = "layer_"+str(index)+"_B"

        return W, B
            

# build submodel
def build_submodel(image_size,
                   num_channels,
                   L_dim_conv_layers,
                   L_filter_size,
                   L_pool_size,
                   L_activation_conv,
                   L_dim_full_layers,
                   L_activation_full,
                   dropout,
                   prediction,
                   allow_comment=False,
		   sub_dropout=0,
		   L_pool_step=[],
		   L_pool_padding=[]):
                   
    # CONVOLUTION
    params_channels = [10**(-i) for i in range(len(L_dim_conv_layers) + 1)]
    index_params = 0
    params_channels.reverse()
    output_dim = num_channels*np.prod(image_size)
    conv_layers = []
    assert len(L_dim_conv_layers) == len(L_filter_size)
    assert len(L_dim_conv_layers) == len(L_pool_size)
    assert len(L_dim_conv_layers) == len(L_activation_conv)
    if len(L_pool_step)==0:
	L_pool_step = [ (1,1) for i in range(len(L_dim_conv_layers))]
	L_pool_padding = [ (0,0) for i in range(len(L_dim_conv_layers))]
    assert len(L_dim_conv_layers) == len(L_pool_step)
    assert len(L_dim_conv_layers) == len(L_pool_padding)
    L_conv_dropout = [dropout]*len(L_dim_conv_layers) # unique value of dropout for now
    convnet = None
    mlp = None
    if len(L_dim_conv_layers):
        for (num_filters, filter_size,
            pool_size, activation_str,
            dropout, index, step, padding) in zip(L_dim_conv_layers,
                                  L_filter_size,
                                  L_pool_size,
                                  L_activation_conv,
                                  L_conv_dropout,
                                  xrange(len(L_dim_conv_layers)),
				  L_pool_step,
				  L_pool_padding
                                  ):

            # convert filter_size and pool_size in tuple
            filter_size = tuple(filter_size)

            if pool_size is None:
                pool_size = (0,0)
            else:
                pool_size = tuple(pool_size)

            # TO DO : leaky relu
            if activation_str.lower() == 'rectifier':
                activation = Rectifier()
            elif activation_str.lower() == 'tanh':
                activation = Tanh()
            elif activation_str.lower() in ['sigmoid', 'logistic']:
                activation = Logistic()
            elif activation_str.lower() in ['id', 'identity']:
                activation = Identity()
            else:
                raise Exception("unknown activation function : %s", activation_str)

            assert 0.0 <= dropout and dropout < 1.0
            num_filters = num_filters - int(num_filters*dropout)

	    layer_conv = Convolutional(filter_size=filter_size,
                                                num_filters=num_filters,
                                                name="layer_%d" % index,
						weights_init=IsotropicGaussian(0.01),
                                                biases_init=Constant(0.0))
	    conv_layers.append(layer_conv)
	    conv_layers.append(activation)
            index_params+=1
	    if not (pool_size[0] == 0 and pool_size[1] == 0):
		#pool = MaxPooling(pooling_size=pool_size, step=step, padding=padding)
		pool = MaxPooling(pooling_size=pool_size)
		conv_layers.append(pool)

        convnet = ConvolutionalSequence(conv_layers, num_channels=num_channels,
                                    image_size=image_size,
                                    name="conv_section")      
        convnet.push_allocation_config()
        convnet.initialize()
        output_dim = np.prod(convnet.get_dim('output'))

    # MLP
    assert len(L_dim_full_layers) == len(L_activation_full)
    L_full_dropout = [dropout]*len(L_dim_full_layers) # unique value of dropout for now

    # reguarding the batch dropout : the dropout is applied on the filter
    # which is equivalent to the output dimension
    # you have to look at the dropout_rate of the next layer
    # that is why we throw away the first value of L_exo_dropout_full_layers
    pre_dim = output_dim
    if allow_comment:
        print "When constructing the model, the output_dim of the conv section is %d." % output_dim
    activations=[]
    dims=[pre_dim]
    if len(L_dim_full_layers):
        for (dim, activation_str,
            dropout, index) in zip(L_dim_full_layers,
                                  L_activation_full,
                                  L_full_dropout,
                                  range(len(L_dim_conv_layers),
                                        len(L_dim_conv_layers)+ 
                                        len(L_dim_full_layers))
                                   ):
                                          
                # TO DO : leaky relu
                if activation_str.lower() == 'rectifier':
                    activation = Rectifier().apply
                elif activation_str.lower() == 'tanh':
                    activation = Tanh().apply
                elif activation_str.lower() in ['sigmoid', 'logistic']:
                    activation = Logistic().apply
                elif activation_str.lower() in ['id', 'identity']:
                    activation = Identity().apply
                else:
                    raise Exception("unknown activation function : %s", activation_str)
                activations.append(activation)
                assert 0.0 <= dropout and dropout < 1.0
                dim = dim - int(dim*dropout)
                if allow_comment:
                    print "When constructing the fully-connected section, we apply dropout %f to add an MLP going from pre_dim %d to dim %d." % (dropout, pre_dim, dim)
                dims.append(dim)
        #now construct the full MLP in one pass:

    activations.append(Identity())
    #params_channels[index_params]
    dims.append(prediction)
    mlp = MLP(activations=activations, dims=dims,
              weights_init=IsotropicGaussian(0.1),
              biases_init=Constant(0.0),
              name="layer_%d" % index)
    mlp.push_allocation_config()
    mlp.initialize()
    return (convnet, mlp)
