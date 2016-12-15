# -*- coding: utf-8 -*-
# 04/03/2016
# @author = Mélanie Ducoffe
# comité formé par batchwise dropout

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import lazy
from blocks.bricks.conv import Flattener
from composite_architecture_new import CompositeSequence, build_submodel
from dropout import retrieve_params
import numpy as np

# objet qui stocke un ConvolutionalSequence + Flattener + MLP

class Architecture(Sequence, Initializable, Feedforward):
    """DOC TO DO
    """

    @lazy(allocation=['num_channels', 'image_size',
                      'L_dim_conv_layers', 'L_filter_size',
                      'L_pool_size', 'L_activation_conv',
                      'L_dim_full_layers', 'L_activation_full',
                      'prediction', 'prob_dropout', 'comitee_size', 'dropout_training'])
    def __init__(self, num_channels, image_size,
                 L_dim_conv_layers, L_filter_size,
                 L_pool_size, L_activation_conv,
                 L_dim_full_layers, L_activation_full,
                 prediction, prob_dropout, comitee_size,
                 dropout_training,
		 L_pool_step=[], L_pool_padding=[],
                **kwargs):
        self.image_size = image_size
        self.num_channels = num_channels
        self.L_dim_conv_layers = L_dim_conv_layers
        self.L_filter_size = L_filter_size
        self.L_pool_size = L_pool_size
        self.L_activation_conv = L_activation_conv
        self.L_dim_full_layers = L_dim_full_layers
        self.L_activation_full = L_activation_full
        self.prediction = prediction
        self.dropout_training = dropout_training
	self.L_pool_step = L_pool_step
	self.L_pool_padding = L_pool_padding
        self.T = 1.
	self.params_savings={}
        convnet, mlp = build_submodel(image_size=self.image_size,
                                      num_channels=self.num_channels,
                                      L_dim_conv_layers=self.L_dim_conv_layers,
                                      L_filter_size=self.L_filter_size,
                                      L_pool_size=self.L_pool_size,
                                      L_activation_conv=self.L_activation_conv,
                                      L_dim_full_layers=self.L_dim_full_layers,
                                      L_activation_full=self.L_activation_full,
                                      dropout=0.,
                                      prediction=self.prediction,
                                      allow_comment=False,
                                      sub_dropout=dropout_training,
				      L_pool_step=self.L_pool_step,
				      L_pool_padding=self.L_pool_padding);

        self.instance = CompositeSequence(convnet, mlp);
        application_methods = [self.instance.apply]
        super(Architecture, self).__init__(
            application_methods=application_methods, **kwargs)

        #comitee
        assert prob_dropout >=0. and prob_dropout <1.
        self.prob_dropout = prob_dropout
        self.comitee_size = comitee_size
	self.params_savings_init={}

    def confidence(self, input_):
	return self.instance.confidence(input_)

    def set_T(self, newT):
        #assert newT >=1.
        self.T = newT
        
    def get_Params(self):
        A, B = self.instance.get_Params();
        return A + B

    def nb_parameters(self):
	params = self.get_Params()
	N = 0
	for p in params:
		n=0
		shape = p.shape.eval()
		N += np.prod(shape)
	return N

    def save_model(self):
	params = self.get_Params()
	for p in params:
		self.params_savings[p.name]=p.get_value()

    def load_model(self):
	params = self.get_Params()
	for p in params:
		if not p.name in self.params_savings.keys():
			raise Exception('unknow key %s', p.name)
		p.set_value(self.params_savings[p.name])


    def get_dim(self, name):
	return self.instance.get_dim(name)
        if name == 'input_':
            return ((self.num_channels,) + self.image_size)
        if name == 'output':
            return self.layers[-1].get_dim(name)
        return super(Architecture, self).get_dim(name)

    def _push_allocation_config(self):
        self.instance._push_allocation_config()

    def generate_instance(self, retrieve=True):
        convnet, mlp = build_submodel(image_size=self.image_size,
                                      num_channels=self.num_channels,
                                      L_dim_conv_layers=self.L_dim_conv_layers,
                                      L_filter_size=self.L_filter_size,
                                      L_pool_size=self.L_pool_size,
                                      L_activation_conv=self.L_activation_conv,
                                      L_dim_full_layers=self.L_dim_full_layers,
                                      L_activation_full=self.L_activation_full,
                                      dropout=self.prob_dropout,
                                      prediction=self.prediction,
				      allow_comment=False,
                                      sub_dropout=0.,
				      L_pool_step=self.L_pool_step,
				      L_pool_padding=self.L_pool_padding);


        comitee_member = CompositeSequence(convnet, mlp);
        comitee_member.initialize()

        if retrieve:
          retrieve_params(self.instance, comitee_member, self.prob_dropout, self.dropout_training)

        return comitee_member


    def dropout_evaluation(self):
      temp_prob_dropout = self.prob_dropout
      temp_dropout_training = self.dropout_training
      self.prob_dropout = 0
      self.dropout_training=0
      instance = self.generate_instance(retrieve=False)
      retrieve_params(self.instance, instance, 0., temp_dropout_training)
      self.prob_dropout = temp_prob_dropout
      self.dropout_training = temp_dropout_training
      return instance

        
    def get_name(self):
        self.instance.get_Params()

    def generate_comitee(self):
        comitee = []
        for i in xrange(self.comitee_size):
            comitee.append(self.generate_instance())
        return comitee

    def save(self):
      dict_params = {}
      params = self.get_Params()
      for w in params:
        assert not w.name in dict_params.keys(), "Screw you !!!"
        dict_params[w.name] = w.get_value()

      return dict_params

    def load(self, dict_params):
      params = self.get_Params()
      for w in params:
        w.set_value(dict_params[w.name])

    def save_file(self, filename):
      dict_params = save();

    def initial_state(self):
	self.instance.initialize()
	params = self.get_Params()
	if len(self.params_savings_init)==0:
		params = self.get_Params()
		for p in params:
			self.params_savings_init[p.name]=p.get_value()
			#print p.name, np.min(p.get_value()), np.max(p.get_value())
	else:
		for p in params:
			if not p.name in self.params_savings_init.keys():
				raise Exception('unknow key %s', p.name)
			p.set_value(self.params_savings_init[p.name])
	self.save_model()
      



