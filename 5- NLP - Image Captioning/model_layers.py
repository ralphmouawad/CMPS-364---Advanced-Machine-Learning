import numpy as np

## RALPH MOUAWAD 
## CMPS 364 - HW 5

def forward_layer(inp, weights, bias):
    output = np.dot(inp.reshape(inp.shape[0], -1), weights) + bias
    cache_data = (inp, weights, bias) # can use this to get Whx.xt or Whh.ht-1, Why.ht
    return output, cache_data


def backward_layer(grd, cache_data):
    inp, weights, bias = cache_data
    d_inp = np.dot(grd, np.transpose(weights)).reshape(inp.shape)
    dweights = np.dot(np.transpose(inp.reshape(inp.shape[0], -1)), grd)
    dbias = np.sum(grd, axis=0) 
    return d_inp, dweights, dbias 


def foward_propg_step(inp, prev_hidden, input_weights, hidden_weights, bias):
    """
    one step of the model's forward propagation
    """
    next_hidden_state, cache_data = None, None
    ##### YOUR CODE STARTS HERE ####
    z_hx, cache_hx = forward_layer(inp, input_weights, bias) # Whx*x_t + b_h
    z_hh, cache_hh = forward_layer(prev_hidden, hidden_weights, 0) # Whh*h_t-1 (bias was added before)
    z = z_hx + z_hh ## This is Whx*x_t + Whh*h_t-1 + b_h

    next_hidden_state = np.tanh(z) # h_t = tanh(Whx*xt + Whh*ht-1 + bh)
    cache_data = (z, inp, input_weights, prev_hidden, hidden_weights, bias) # take this data to use it in backward
    ##### YOUR CODE ENDS HERE ####
    return next_hidden_state, cache_data


def backward_propg_step(dnext_h_state, cache_data):
    """
    one step of the model's backward propagation
    """
    d_inp, dprev_hidden, dinput_weights, dhidden_weights, dbias = None, None, None, None, None
    ##### YOUR CODE STARTS HERE ####
    # z = cache_data[0]
    # inp = cache_data[1]
    # input_weights = cache_data[2]
    # prev_hidden = cache_data[3]
    # hidden_weights = cache_data[4]
    # bias = cache_data[5]
    z, inp, input_weights, prev_hidden, hidden_weights, bias = cache_data

    dz = dnext_h_state*(1-np.square(np.tanh(z))) # gradient of loss function wrt 'z' using the given grd wrt h_t

    d_inp, dinput_weights, dbias = backward_layer(dz, (inp, input_weights, bias)) # this will compute: dinp = dz*Whx, dinput_weights=dz*xt, dbias=1/ sum
    
    dprev_hidden, dhidden_weights, dbias = backward_layer(dz, (prev_hidden, hidden_weights, bias)) # dprevhidden = dz*ht-1, dhiddenw = ht-1*dz, dbias computed again. 
    ##### YOUR CODE ENDS HERE ####
    return d_inp, dprev_hidden, dinput_weights, dhidden_weights, dbias


def foward_propg_rnn(inp, init_hidden_state, input_weights, hidden_weights, bias): ## here, inp is 3D where we have 2 captions, each of 3 words, and every word 10 dimensions. I referred to the example in Instruction
    hidden_states, cache_data = None, None
    ##### YOUR CODE STARTS HERE ####
    cache_data = []
    hidden_states = np.zeros((inp.shape[0], inp.shape[1], input_weights.shape[1]))
    prev_hidden = np.copy(init_hidden_state) # initialize it to update later
    # for i in range(inp.shape[0]): 
    # # I could have done it also with 1 for loop. Im doing it like this to make sure everything is processes good
    #     for t in range(inp.shape[1]):

    #         x_t = inp[i, t, :] # word embedding at time 't' for sample 'i'
    #         x_t = x_t.reshape(1,-1)
    #         next_hidden_state, cache = foward_propg_step(x_t, prev_hidden[i,:].reshape(1,-1), input_weights, hidden_weights, bias) # there is a typo foward instead of forward
    #         hidden_states[i,t,:] = next_hidden_state.squeeze()
    #         cache_data[(i,t)] = cache
    #         prev_hidden[i,:] = next_hidden_state.squeeze()

    ### IMPORTANT Note
    ### The first code works, but I faced issues loading cache data at timestep t for backward as it was saved before for every sample & timestep.
    ### I did this again with one for loop as the samples are treated simultaneously, it's more efficient
    ### and fixed the issue from my second code. However, it is the same way my prev code was working

    for t in range(inp.shape[1]):
        x_t = inp[:, t, :] # word embedding at time 't' for sample 'i'
        # x_t = x_t.reshape(1,-1)
        next_hidden_state, cache = foward_propg_step(x_t, prev_hidden, input_weights, hidden_weights, bias) # there is a typo foward instead of forward
        hidden_states[:,t,:] = next_hidden_state.squeeze()
        # prev_hidden[i,:] = next_hidden_state.squeeze()
        prev_hidden=next_hidden_state.squeeze()
        cache_data.append(cache)

    ##### YOUR CODE ENDS HERE ####
    return hidden_states, cache_data

def backward_propg_rnn(dh_states, cache_data): 
    d_inp, dinit_h_state, dinput_weights, dhidden_weights, dbias = None, None, None, None, None
    (batch_size, seqlen, H) = dh_states.shape # I want the length of word features
    D = cache_data[0][1].shape[1]
    ##### YOUR CODE STARTS HERE ####
    d_inp = np.zeros((batch_size, seqlen, D))
    dinit_h_state= np.zeros((batch_size, H))
    dinput_weights = np.zeros((D, H))
    dhidden_weights = np.zeros((H,H))
    dbias = np.zeros(H)

    dh_next = np.zeros((batch_size, H))
    # #backpropagation through time, we have to move backward to compute the gradients
    # for t in reversed(range(seqlen)): #through each time step
    #     for i in range(batch_size): # through each sample
    #         cache=cache_data([i,t])
            
    #         dx, dprev_hidden, dw_inp, dw_h, db = backward_propg_step(dh_next[i,:], cache)
            
    #         d_inp[i,t,:]=dx
    #         dinput_weights+=dw_inp
    #         dhidden_weights+=dw_h
    #         dbias+=db
    #         dh_next[i]=dprev_hidden
    # dinit_h_state=dh_next

    # the 2 for loops were giving errors bcz of matrix operation.
    # I will only iterate over t as every sample can still be treated alone
    # I replaced forward_rnn with one loop to fix cache_data per timestep. 
    for t in reversed(range(seqlen)):
        # cache = []
        # # retrieve the data
        # for i in range(batch_size):
        #     cache.append(cache_data[(i,t)]) #will use this to retrieve the data over every time step for all samples
        cache=cache_data[t]

        dh_next += dh_states[:,t,:] #get grad of ht across all samples
        dx, d_prevh, dw_inp, dw_h, db = backward_propg_step(dh_next, cache) #compute the gradients
     # sum the gradients over all samples and time
        d_inp[:,t,:]= dx
        dinput_weights += dw_inp #the gradients will acc over time for the weights
        dhidden_weights += dw_h
        dbias += db
        dh_next=d_prevh #here we pass the last grad to the current step bcz it is backward prop
    dinit_h_state=dh_next
    # ##### YOUR CODE ENDS HERE ####
    return d_inp, dinit_h_state, dinput_weights, dhidden_weights, dbias


def forward_word_embedding(x, W):
    out, cache_data = None, None
    ##### YOUR CODE STARTS HERE ####
    out = np.array([W[i] for i in x]) 
    ## if the word index is 3, then go to the 3rd row of embedding matrix and retrieve the feature vector
    cache_data = (x, W)
    ##### YOUR CODE ENDS HERE ####
    return out, cache_data


def backward_word_embedding(dev_out, cache_data):
    dWeights = None
    ##### YOUR CODE STARTS HERE ####
    x, W = cache_data
    dWeights = np.zeros_like(W)
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            word_id = x[i, t] # retrieve the word at this time
            dWeights[word_id] += dev_out[i,t] # add the gradient to the word retrievedd
    ##### YOUR CODE STARTS HERE ####
    ##### YOUR CODE ENDS HERE ####
    return dWeights


def sigmoid(x):
    """
    Numerically stable implementation of the logistic sigmoid function to prevent overflow or underflow.

    This implementation avoids potential issues with large or small values of x by splitting the
    computation into two parts based on the sign of x. For positive values of x, the standard
    formula is used, while for negative values of x, the computation is adjusted for stability.

    Parameters:
    - x: A NumPy array of input values.

    Returns:
    - The sigmoid of each element in x, calculated in a numerically stable way.
    """
    # Mask for positive values of x
    pos_mask = x >= 0

    # Mask for negative values of x
    neg_mask = x < 0

    # Initialize an array for intermediate results
    z = np.zeros_like(x)

    # For positive x, compute exp(-x)
    z[pos_mask] = np.exp(-x[pos_mask])

    # For negative x, compute exp(x)
    z[neg_mask] = np.exp(x[neg_mask])

    # Initialize the result array with ones
    top = np.ones_like(x)

    # For negative x, set top to the computed z values
    top[neg_mask] = z[neg_mask]

    # Return the final sigmoid value: 1 / (1 + exp(-x)) for positive x
    # and 1 / (1 + exp(x)) for negative x, computed in a numerically stable way.
    return top / (1 + z)

def foward_propg_step_lstm(inp, prev_hidden, prev_cell_state, input_weights, hidden_weights, bias):
    next_hidden_state, next_cell_state, cache_data = None, None, None
    ##### YOUR CODE STARTS HERE ####
    # here I'll do the forward w/o the prev function bcz it has to be transpose
    a = np.dot(inp, input_weights) + np.dot(prev_hidden, hidden_weights) + bias
    ai, af, ao, ag = np.hsplit(a, 4) # this hint was given in the instructions
    i= sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g= np.tanh(ag)

    next_cell_state = f*prev_cell_state + i*g
    next_hidden_state = o*np.tanh(next_cell_state)

    cache_data = (prev_hidden, prev_cell_state, inp, input_weights, hidden_weights, next_cell_state, bias, i, f, o, g)
    ##### YOUR CODE ENDS HERE ####
    return next_hidden_state, next_cell_state, cache_data


def backwd_propg_step_lstm(dnext_h_state, dnext_cell_state, cache_data):
    ## I referred to this article to get the gradients: https://towardsdatascience.com/lstm-gradients-b3996e6a0296
    d_inp, dprev_hidden, dprev_cell_state, dinput_weights, dhidden_weights, dbias = None, None, None, None, None, None
    ##### YOUR CODE STARTS HERE ####
    prev_hidden, prev_cell_state, inp, input_weights, hidden_weights, next_cell_state, bias, i, f, o, g = cache_data # we'll computhe the gradient wrt each term. check my pdf where I wrote each one

    dnext_cell_state+=dnext_h_state*o*(1-np.tanh(next_cell_state)**2)
    di = dnext_cell_state*g
    df= dnext_cell_state*prev_cell_state
    dg=dnext_cell_state*i
    do=dnext_h_state * np.tanh(next_cell_state)
    dprev_cell_state= dnext_cell_state*f

    ## gradient through ai,af... before activations
    d_ai=di*i*(1-i)
    d_af=df*f*(1-f)
    d_ag=dg*(1-np.square(g))
    d_ao=do*o*(1-o)
    da= np.hstack((d_ai, d_af, d_ao, d_ag))

    ## gradient wrt input, weights...
    ## I could have also used forward_layer but I wil compute them by hand to make no mistake
    ## I referred to how the matrices were put in forward_layer. 
    dprev_hidden= np.dot(da, hidden_weights.T)
    dinput_weights=np.dot(inp.T, da)
    d_inp= np.dot(da, input_weights.T)
    dhidden_weights= np.dot(prev_hidden.T, da)
    dbias = np.sum(da, axis=0)

    ##### YOUR CODE ENDS HERE ####
    return d_inp, dprev_hidden, dprev_cell_state, dinput_weights, dhidden_weights, dbias


def forward_propg_single_lstm(inp, init_hidden_state, input_weights, hidden_weights, bias):
    h, cache_data = None, None
    ##### YOUR CODE STARTS HERE ####
    cell= np.zeros_like(init_hidden_state)
    h = np.zeros((inp.shape[0], inp.shape[1], init_hidden_state.shape[1])) 
    cache_data = []
    ## initialize the hidden the states & cell states
    prev_hidden = init_hidden_state
    prev_cell_state= cell
    ### similar implementation to the one with rnn
    ### could have done it with 2 for loops through each sample and each step like forward
    ### but it'd have given me same problem in the backward phase
    for t in range(inp.shape[1]):
        x_t = inp[:,t,:]
        next_hidden_state, next_cell_state, cache = foward_propg_step_lstm(x_t, prev_hidden, prev_cell_state, input_weights, hidden_weights, bias)
        h[:,t,:]= next_hidden_state
        prev_hidden=next_hidden_state
        prev_cell_state=next_cell_state
        cache_data.append(cache)
    ##### YOUR CODE ENDS HERE ####

    return h, cache_data


def backwd_propg_single_lstm(dh, cache_data):
    d_inp, dinit_h_state, dinput_weights, dhidden_weights, dbias = None, None, None, None, None
    ##### YOUR CODE STARTS HERE ####
    ## similar to backward_rnn but we have cell states also
    batch_size, seqlen, H = dh.shape # I want the length of word features # I took this from prev code
    D = cache_data[0][2].shape[1] #word length, from the input shape; check before
    d_inp = np.zeros((batch_size, seqlen, D))
    dinit_h_state= np.zeros((batch_size, H))
    # dinput_weights = np.zeros((D, H))
    # dhidden_weights = np.zeros((H,H)) those aren't working so I'll just initialize them as they are stored in cache data
    # dbias = np.zeros(H)
    dbias= np.zeros_like(cache_data[0][6])
    dinput_weights = np.zeros_like(cache_data[0][3])
    dhidden_weights= np.zeros_like(cache_data[0][4])
    dinit_c= np.zeros((batch_size, H))
    dnext_h= np.zeros((batch_size, H))
    dnext_c= np.zeros_like(dinit_c)
    # d_inp, dprev_hidden, dprev_cell_state, dinput_weights, dhidden_weights, dbias
    for t in reversed(range(seqlen)):
        cache=cache_data[t]

        dnext_h += dh[:,t,:] #get grad of ht across all samples
        dx, dprev_h, dprev_c, dw_inp, dw_h, db = backwd_propg_step_lstm(dnext_h, dnext_c, cache) #compute the gradients
     # sum the gradients over all samples and time
        d_inp[:,t,:]= dx
        dinput_weights += dw_inp #the gradients will acc over time for the weights
        dhidden_weights += dw_h
        dbias += db
        dnext_h=dprev_h #here we pass the last grad to the current step bcz it is backward prop
        dnext_c= dprev_c
    dinit_h_state=dnext_h
    dinit_c= dnext_c ### this can be not returned as we're not returning it. I kept it as I thought it might be needed 
    ##### YOUR CODE ENDS HERE ####
    return d_inp, dinit_h_state, dinput_weights, dhidden_weights, dbias


def temp_forward_layer(inp, weight, bias):
    batch_size, seq_len, input_dim = inp.shape
    output_dim = bias.shape[0]
    output = inp.reshape(batch_size * seq_len, input_dim).dot(weight).reshape(batch_size, seq_len, output_dim) + bias
    cache_data = inp, weight, bias, output
    return output, cache_data


def temp_backward_layer(dev_out, cache_data):
    inp, weight, bias, out = cache_data
    batch_size, seq_len, input_dim = inp.shape
    output_dim = bias.shape[0]

    d_inp = dev_out.reshape(batch_size * seq_len, output_dim).dot(weight.T).reshape(batch_size, seq_len, input_dim)
    dweight = dev_out.reshape(batch_size * seq_len, output_dim).T.dot(inp.reshape(batch_size * seq_len, input_dim)).T
    dbias = dev_out.sum(axis=(0, 1))

    return d_inp, dweight, dbias

def masked_softmax_loss(inp, target, mask, ver=False):

    batch_size , seqlen, vocab = inp.shape

    inp_2d = inp.reshape(batch_size * seqlen,vocab)
    target_2d = target.reshape(batch_size * seqlen)
    mask_2d = mask.reshape(batch_size * seqlen)

    probs = np.exp(inp_2d - np.max(inp_2d, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_2d * np.log(probs[np.arange(batch_size * seqlen), target_2d])) / batch_size
    d_inp_2d = probs.copy()
    d_inp_2d[np.arange(batch_size * seqlen), target_2d] -= 1
    d_inp_2d /= batch_size
    d_inp_2d *= mask_2d[:, None]

    if ver:
        print("d_inp_2d: ", d_inp_2d.shape)

    d_inp = d_inp_2d.reshape(batch_size, seqlen, vocab)

    return loss, d_inp
