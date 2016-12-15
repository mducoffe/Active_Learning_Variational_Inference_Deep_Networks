def build_dictionnary(cost, variables):
	items = []
	conv_index = 0; fully_index=0
	print 'test build dictionnary'
	import pdb
	pdb.set_trace()
	for var in variables:
		if var.name is None:
			continue
		if var.name=="convolutional_apply_input_":
			var_input = var
		if var.name=="convolutional_apply_output":
			var_output = var
			items.append(("conv_input_"+str(conv_index), var_input))
			items.append(("conv_output_"+str(conv_index), var_output))
			conv_index+=1
		if var.name=="linear_"+str(fully_index)+"_apply_input_":
			var_input = var
		if var.name=="linear_"+str(fully_index)+"_apply_output":
			var_output = var
			items.append( ("fully_input_"+str(fully_index), var_input))
			items.append( ("fully_output_"+str(fully_index), var_output))
			fully_index +=1

	dico = OrderedDict(items)

	# for memory consumption forget this step
	# preprocessing
	keys = dico.keys()
	# fully connected layers
	i = 0
	# TO DO : checl that the activation and pre activations are in the right sense
	while "fully_input_"+str(i) in keys:
		# because we a*aT instead of aT*a we took directly the transpose
		[grad_s] = T.grad(cost, [dico["fully_output_"+str(i)]])
		dico["fully_output_"+str(i)] = grad_s
		var_input = dico["fully_input_"+str(i)]
		var_input = T.concatenate([var_input, T.ones((var_input.shape[0], 1))], axis=1)
		dico["fully_input_"+str(i)] = var_input
		i+=1

	# convolutional layers
	j = 0
	while "conv_input_"+str(j) in keys:
		[grad_s] = T.grad(cost, [dico["conv_output_"+str(j)]])
		grad_s = grad_s.dimshuffle((1, 2, 3, 0))
		shape = grad_s.shape
		grad_s = grad_s.reshape((shape[0], shape[1]*shape[2]*shape[3]))
		grad_s = grad_s.dimshuffle((1,0)) # transpose to satisfy Roger inequality
		dico["conv_output_"+str(j)] = grad_s
		j+=1
	return dico



