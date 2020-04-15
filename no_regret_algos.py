def normalize_list(L):
	''' takes in a list of numbers L and returns a version of L where all entries sum to 1 '''

	return [e/sum(L) for e in L]

def update_weights(prev_wts, loss, eta, rule='mw'):
	'''
		takes in a list of weights at the previous time step, the 
		loss for each action at the current time step, and a learning 
		rate eta and performs a single step of the specified algorithm

		returns the updated list of weights 
	'''

	# various checks

	# update this as we implement more of these
	if rule != 'mw':
		print("only MW implemented, returning previous weights")
		return prev_wts

	if len(prev_wts) != len(loss):
		print("The number of weights you've given is {} but  the number of losses is {}".format(
			len(prev_wts),len(loss)))
		return prev_wts

	return [  w * (1 - l) for w, l in zip(prev_wts, loss)     ]



def run_mult_weights(loss_vectors, eta, rule='mw', initial_weights = None, return_history = False):
	'''
		takes in a list of lists of losses, a learning rate eta, and a rule 
		and repeatedly calls `update_weights` to run the specified flavor of 
		multiplicative weights.  initial weights can be specifed, or default to 
		uniform. setting return_history to True returns the entire history of 
		weights, otherwise just returns the weights at the last step
	'''

	# data sanity checks go here

	history = []

	if initial_weights == None: 
		wts = [1 for e in loss_vectors[0]]

	else:
		wts = normalize_list(initial_weights)

	history.append(wts)

	for loss in loss_vectors:
		wts = update_weights(wts, loss, eta, rule)
		wts = normalize_list(wts)
		history.append(wts)

	if return_history:
		return history
	else:
		return wts
