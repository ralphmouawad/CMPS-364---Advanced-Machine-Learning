
## RALPH MOUAWAD
import numpy as np 
from model_layers import *
class ImageCaptioningRNN:
    """
    The ImageCaptioningRNN model generates captions for images by processing
    image feature vectors through a recurrent neural network (RNN).

    This model takes input image feature vectors of dimension `D` and processes them
    using an RNN with a hidden state of size `H`. It outputs word vectors of size `W`
    over sequences of length `seqlen`, using a vocabulary with `vocab` unique words.
    Training is performed in minibatches of size `batch_size`.

    Note: This implementation does not include any regularization techniques.
    """

    def __init__(
            self,
            word_to_idx,
            input_dim=512,
            wordvec_dim=128,
            hidden_dim=128,
            cell_type="rnn",
            dtype=np.float32,
        ):
            """
            Initializes the ImageCaptioningRNN model with the provided configuration.

            Parameters:
            - word_to_idx: A dictionary mapping words in the vocabulary to unique integer indices.
            - input_dim: The dimensionality `D` of the input image feature vectors.
            - wordvec_dim: The dimensionality `W` of word vectors used to represent vocabulary words.
            - hidden_dim: The size `H` of the RNN's hidden state.
            - cell_type: Specifies the type of RNN cell, either 'rnn' (vanilla RNN) or 'lstm'.
            - dtype: Specifies the data type for parameters and computations (e.g., `np.float32` for
            training or `np.float64` for gradient checking).
            """
            # Ensure the provided RNN cell type is valid
            if cell_type not in {"rnn", "lstm"}:
                raise ValueError(f'Invalid cell_type "{cell_type}". Expected "rnn" or "lstm".')

            # Initialize key model attributes
            self.cell_type = cell_type
            self.dtype = dtype
            self.word_to_idx = word_to_idx
            self.index_to_word = {i: word for word, i in word_to_idx.items()}  # Reverse the word-to-index mapping
            self.params = {}

            vocab_size = len(word_to_idx)

            # Identify special tokens for padding, start, and end of sequences
            self._null = word_to_idx["<NULL>"]
            self._start = word_to_idx.get("<START>", None)
            self._end = word_to_idx.get("<END>", None)

            # Initialize word embedding matrix (shape: vocab_size x wordvec_dim)
            self.params["W_embedding"] = np.random.randn(vocab_size, wordvec_dim) / 100

            # Initialize projection matrix and bias for mapping image features to hidden state space
            self.params["W_project"] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
            self.params["b_proj"] = np.zeros(hidden_dim)

            # Set up RNN-specific parameters
            dim_multiplier = 4 if cell_type == "lstm" else 1  # LSTM requires 4 times the size for gates
            self.params["input_weights"] = np.random.randn(wordvec_dim, dim_multiplier * hidden_dim) / np.sqrt(wordvec_dim)
            self.params["hidden_weights"] = np.random.randn(hidden_dim, dim_multiplier * hidden_dim) / np.sqrt(hidden_dim)
            self.params["bias"] = np.zeros(dim_multiplier * hidden_dim)

            # Initialize weights and bias for the output layer mapping hidden states to vocabulary scores
            self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size) / np.sqrt(hidden_dim)
            self.params["b_vocab"] = np.zeros(vocab_size)

            # Convert all parameters to the specified data type
            for param_name, param_value in self.params.items():
                self.params[param_name] = param_value.astype(self.dtype)


    def loss(self, image_features, ground_truth_captions):
        # Prepare input and output sequences.
        # captions_input: Contains all words except the last word; it is used as the input for the model
        # captions_target: Contains all words except the first word; it is the target output for the model
        captions_input = ground_truth_captions[:, :-1]  # Exclude the last word for input
        captions_target = ground_truth_captions[:, 1:]  # Exclude the first word for target output

        # Create a mask to ignore padding (NULL tokens) during loss computation
        mask = captions_target != self._null  # Mask out <NULL> tokens for loss calculation

        # Parameters for the affine transform from image image_features to the initial hidden state of the model
        W_project, b_proj = self.params["W_project"], self.params["b_proj"]

        # Word embedding matrix, which maps words to their vector representations
        W_embedding = self.params["W_embedding"]

        # Model parameters: input-to-hidden weights, hidden-to-hidden weights, and bias
        input_weights, hidden_weights, bias = self.params["input_weights"], self.params["hidden_weights"], self.params["bias"]

        # Parameters for the final transformation from hidden state to vocabulary space (output layer)
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        # Initialize loss and gradients (gradients will accumulate here)
        loss, grads = 0.0, {}

        ############################################################################
        # Image Captioning Forward Pass Implementation Details:                    #
        #                                                                          #
        # This forward pass will transform image features into a descriptive       #
        # caption by following these key steps:                                    #
        #                                                                          #
        # 1. Initial State Creation:                                               #
        #    - Transform raw image features into an initial hidden state           #
        #    - Ensure the output is a matrix of shape (batch_size, H) where:       #
        #      batch_size = number of images, H = hidden state dimension           #
        #                                                                          #
        # 2. Word Embedding Transformation:                                        #
        #    - Convert input word indices to dense vector representations          #
        #    - Produce an embedding matrix of shape (batch_size, seqlen, W) where: #
        #      seqlen = sequence length, W = word embedding dimension              #
        #                                                                          #
        # 3. Sequence Processing:                                                  #
        #    - Use either RNN or LSTM to process word vectors                      #
        #    - Generate hidden state vectors for each timestep                     #
        #    - Output will be a 3D tensor of shape (batch_size, seqlen, H)         #
        #                                                                          #
        # 4. Vocabulary Scoring:                                                   #
        #    - Apply a temporal affine transformation to compute                   #
        #      vocabulary scores at each timestep                                  #
        #    - Resulting scores will be in shape (batch_size, seqlen, vocab)       #
        #      where vocab = vocabulary size                                       #
        #                                                                          #
        # 5. Loss Computation:                                                     #
        #    - Use temporal softmax to calculate loss                              #
        #    - Ignore padding/NULL tokens during loss calculation                  #
        #                                                                          #
        # Backward Pass Requirements:                                              #
        #    - Compute gradients for ALL model parameters                          #
        #    - Store loss in 'loss' variable                                       #
        #    - Store parameter gradients in 'grads' dictionary                     #
        #                                                                          #
        # Helpful Hint: Feel free to leverage helper functions from model_ayers.py #
        ############################################################################

        ### YOUR CODE STARTS HERE ###

        # 1- Transofrm initial input vector to an initial hidden state
        # Wproj and bproj are the ones cited before that should be used to transform to h0
        init_h_state= np.dot(image_features, W_project)+b_proj # W*x+bias -- size is (batchsize, Hiddensize)

        # 2- to embed word into their vectors, I will use the W.E. fct to map them into their corresponding vectors
        WE, WE_cache= forward_word_embedding(captions_input, W_embedding) ## cap_input contains the indices, and W_embedding is the containing the word representations that we aim to map each word to

        # 3- I'll check the cell_type that was initialized before to see if I have RNN or LSTM to use
        if self.cell_type=='rnn': #that was stated at the beginning
            # generate hidden states at each timme step
            h_t, cache_rnn= foward_propg_rnn(WE, init_h_state, input_weights, hidden_weights, bias) # the input here are the words rep by vectors. We're generating the hidden states using the RNN
        
        if self.cell_type== 'lstm':
            # init_c_state= np.zeros_like(init_h_state)
            h_t, cache_lstm = forward_propg_single_lstm(WE, init_h_state, input_weights, hidden_weights, bias)

        # 4- Vocab Scoring using affine functions
        output_temp, cache_temp= temp_forward_layer(h_t, W_vocab, b_vocab) # use the parameters specified above for tem forward
        # 5- Loss computation using softmax
        loss, dout_temp= masked_softmax_loss(output_temp, captions_target, mask) # get the loss btw our results and the expected captions. 

        ### Backward Propagation for all models
        # we first need to retrieve the gradient wrt h_t using backward_temp
        dh_t, dW_vocab, db_vocab= temp_backward_layer(dout_temp, cache_temp) #now we can get the gradients wrt the other parameters

        # check if we are using rnn
        if self.cell_type=='rnn':
            d_WE, dinit_h_state, dinput_weights, dhidden_weights, dbias= backward_propg_rnn(dh_t, cache_rnn)
        # check if the layer is LSTM
        if self.cell_type=='lstm':
            d_WE, dinit_h_state, dinput_weights, dhidden_weights, dbias= backwd_propg_single_lstm(dh_t, cache_lstm)
        dW_embedding= backward_word_embedding(d_WE, WE_cache)
        dW_project= np.dot(image_features.T, dinit_h_state)
        db_proj= np.sum(dinit_h_state) ## I got this from the first codes in model_layers. To compute these gradients

        ### Store all gradients inside grad:
        grads['W_vocab']= dW_vocab
        grads['b_vocab']= db_vocab
        grads['W_embedding']= dW_embedding
        # grads['init_h_state']= dinit_h_state 
        grads['hidden_weights']= dhidden_weights
        grads['input_weights']= dinput_weights
        grads['bias']= dbias
        grads['W_project']= dW_project
        grads['b_proj']= db_proj




        ### YOUR CODE ENDS HERE ###

        return loss, grads

    def sample(self, image_features, max_len=30):
        # Number of images in the batch (batch_size)
        batch_size = image_features.shape[0]

        # Initialize caption labels array with the <NULL> token, which will be updated during sampling
        ground_truth_captions = self._null * np.ones((batch_size, max_len), dtype=np.int32)

        # Unpack model parameters
        W_project, b_proj = self.params["W_project"], self.params["b_proj"]  # Projection from image features to hidden state
        W_embedding = self.params["W_embedding"]  # Word embedding matrix
        input_weights, hidden_weights, bias = self.params["input_weights"], self.params["hidden_weights"], self.params["bias"]  # RNN weights
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]  # Weights for transforming hidden state to vocab space

        ###########################################################################
        # Image Caption Generation (Test-Time Sampling) Implementation Guide      #
        #                                                                         #
        # Goal: Generate a caption for a given image by sequentially              #
        # predicting the most likely word at each timestep.                       #
        #                                                                         #
        # Initialization Phase:                                                   #
        # - Transform input image features into an initial hidden state           #
        #   using the learned affine transformation                                #
        #                                                                         #
        # Caption Generation Steps (for each timestep):                           #
        # 1. Word Embedding:                                                      #
        #    - Convert the previous word into its dense vector representation     #
        #    - Use pre-trained word embedding weights                             #
        #                                                                         #
        # 2. RNN State Update:                                                    #
        #    - Use the previous hidden state and current word embedding           #
        #    - Compute the next hidden state via RNN step                         #
        #                                                                         #
        # 3. Vocabulary Scoring:                                                  #
        #    - Apply affine transformation to current hidden state                 #
        #    - Generate score for each word in vocabulary                         #
        #                                                                         #
        # 4. Word Selection:                                                      #
        #    - Choose the word with the highest score                             #
        #    - Add selected word index to output caption                          #
        #                                                                         #
        # Starting Conditions:                                                    #
        # - Begin generation with special <START> token                           #
        #                                                                         #
        # Implementation Notes:                                                   #
        # - Works with minibatch processing                                       #
        # - For LSTM: Initialize first cell state as zeros                         #
        # - No mandatory stopping at <END> token (optional)                       #
        #                                                                         #
        # Performance Tip: Use single-step RNN/LSTM forward functions             #
        # in a loop, NOT the full sequence forward functions                      #
        ###########################################################################

        ### YOUR CODE STARTS HERE ###
        ## Initialize by computing hidden state with the weights we already learned
        h= np.dot(image_features, W_project)+b_proj
        if self.cell_type=='lstm':
                c=np.zeros_like(h)
        ## 1) Word Embedding
        word= np.full(image_features.shape[0], self._start).astype(int) ## batchsize, 

        for i in range(max_len): ## here it is 30
            WE= W_embedding[word] ## Pre-trained word embedding
            ## 2) check the layer type and apply function
            if self.cell_type=='rnn':
                h, cache_rnn= foward_propg_step(WE, h, input_weights, hidden_weights, bias)
            if self.cell_type=='lstm':
                # c=np.zeros_like(h) add this only at the beginning 
                h, c, cache_lstm= foward_propg_step_lstm(WE, h, c, input_weights, hidden_weights, bias)
            ## 3) Vocabulary Scoring using affine transformation
            vocabulary_scoring= np.dot(h, W_vocab)+b_vocab # here the weights are pre-trained. I didnt the function temp... bcz of errors in matrix size
            word= np.argmax(vocabulary_scoring, axis=1)
            ground_truth_captions[:,i]= word  ## 4- word selection
            ## 
        ### YOUR CODE ENDS HERE ###
        return ground_truth_captions
