######### TEST BATCH NORM AND BATCH DROPOUT #########
#### @author Melanie Ducoffe ########################
#### @date 04/03/2016 ###############################
#####################################################
from architecture_new import Architecture
import theano.tensor as T
import theano
import numpy as np

def test_build_submodel():
	# build an architecture
	print 'TESTING BUILD SUBMODEL WITH BATCH NORMALIZATION'
	num_channels=3
	image_size=(32,32)
	L_dim_conv_layers=[13, 15]
	L_filter_size=[(3,3), (3,3)]
	L_pool_size=[(2,2), (2,2)]
	L_activation_conv=['rectifier', 'rectifier']
	L_dim_full_layers=[22]
	L_activation_full=['rectifier']
	prediction=10
	prob_dropout=0.5
	comitee_size=1
	dropout_training=0
	
	model = Architecture(num_channels=num_channels,
			     image_size=image_size,
			     L_dim_conv_layers=L_dim_conv_layers,
			     L_filter_size=L_filter_size,
			     L_pool_size=L_pool_size,
			     L_activation_conv=L_activation_conv,
			     L_dim_full_layers=L_dim_full_layers,
			     L_activation_full=L_activation_full,
			     prediction=prediction,
			     prob_dropout=prob_dropout,
			     comitee_size=comitee_size,
			     dropout_training=dropout_training)

	model.initialize()
	#f = theano.function([x], y, allow_input_downcast=True)
	#x_value = np.random.ranf((1, 3, 32, 32))
	#y_value = f(x_value)

	committee = model.generate_comitee()
	instance, = committee

	full_params = model.get_Params()
	sub_params_W, sub_params_B = instance.get_Params()
	sub_params = sub_params_W+sub_params_B

	for f,s in zip(full_params, sub_params):
		print (f.name, f.shape.eval(), s.shape.eval())
	x = T.tensor4('x')
	y = instance.apply(x)
	f = theano.function([x], y, allow_input_downcast=True)
	x_value = np.random.ranf((1, 3, 32, 32))
	y_value = f(x_value)
	import pdb
	pdb.set_trace()

if __name__=="__main__":
	test_build_submodel()
