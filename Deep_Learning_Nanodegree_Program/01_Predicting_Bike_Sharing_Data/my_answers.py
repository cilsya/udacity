import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        #
        # NOTE: mx + b, looks like b is zero in this case.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        #
        # Activation function
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        #
        # It is stated in the notebook that the activation function for
        # the output layer isAnimCurve f(x) = x. 
        # Basically, this means that the output node is the same as the
        # input node (which they said explicitly as well.)
        #
        # NOTE: This is not efficient to set a variable to another to just
        #       return it in the next line but I am leaving it for clarity.
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(error,self.weights_hidden_to_output.T)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error
        
        hidden_error_term = hidden_error * hidden_outputs*(1 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term*X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term*hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr*(delta_weights_h_o/n_records) # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr*(delta_weights_i_h/n_records) # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################

# NOTE: My strategy for hyperparameter tuning
#
# Step 0101 - In general, change ONE VARIABLE AT A TIME. This makes for a 
#             simple A/B comparison between simulations to diagnose what is 
#             affecting the solve and is it getting better or worse.
#
# Step 0201- Start with number of hidden nodes between the number of input 
#            and output nodes. (In this case, the number of input nodes is 56, I checked,
#            so I hard coded it. If this was a real program, I would dynamically figure it out.)
#            
#            I'll probably choose halfway to get started (i.e. 28,) and adjust it again last
#            after the other hyperparameters.
# 
# Step 0301 - Next I'll adjust the number of iterations. I'll go up by multiples of 10
#             (i.e. 10, 100, 1000, 10000)
#             I will look at the graph and see mainly around what iteration step the validation loss stops 
#             generally descending; or if it whacks out on either training loss or validation loss. If it 
#             to be smaller and try again.
#             
#             When I have the graph not whacking out for the training loss or the validation loss,
#             and the training loss is low and the validation loss pretty much stagnant at a low number,
#             I will set the iteration at that number. Setting the iteration more would be a waste of time
#             because it is validation loss number is not getting better.
#
# Step 0401 - I will try smaller learning rates, this will make training longer (as well as increasing the 
#             number of hidden nodes or increasing the number of iterations but I find this affects the 
#             speed more when compared to the other hyperparameters because I have general guidelines 
#             to stay within for the other ones.) The guideline for this is to try to have as a starting
#             point 1 / n_records. (number of training examples.) I checked in this example
#             so I will hard-code 
#             train_features.shape[0]: 15435
#
#             NOTE: This number seems really small to me. I am wondering if n_records is supposed to mean
#                   number of features and not number of training examples but the code comment in the
#                   assignment 
#
#                   # Go through a random batch of 128 records from the training data set
#
#                   makes me believe it it the number of training examples.
#
# Step 0501 - I come back and try to adjust the number of hidden layers. I set it at the beginning to 
#             be in the ball park (at least, theoretically.) I will try the extremes (1 node, then the
#             number of input nodes.) Then I will try the half rule, (i.e. half way between, then half
#             of that, etc...)


# TARGET:
# The training loss is below 0.09 
# and the validation loss is below 0.18.
#
# NOTE:
# You'll generally find that the best number of hidden nodes to use ends 
# up being between the number of input and output nodes.
# Also,  the number of hidden units should not be more than the twice of 
# the number of input units
# Checked N_i = train_features.shape[1]
# result: 56
#
# The default
# iterations = 100
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1

#
# NOTE: if you effectively divide the learning 
#       rate by n_records, try starting out with a 
#       learning rate of 1
#learning_rate = learning_rate / 15435

# Run: 0005
#
# Note: From graph, lookis like there is an abrupt stop around 1000 iteraions.
#       Going to stick with 1000 for now.
# iterations = 10000
# learning_rate = 0.1
# hidden_nodes = 28
# output_nodes = 1
#


# iterations = 5000
# learning_rate = 0.5
# hidden_nodes = 70
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.062 ... Validation loss: 0.174 ... iter: 7500 ... learn: 1.5 ... hidden: 10
# iterations = 7500
# learning_rate = 1.5
# hidden_nodes = 10
# output_nodes = 1

# #Progress: 100.0% ... Training loss: 0.078 ... Validation loss: 0.149 ... iter: 2000 ... learn: 1.0 ... hidden: 10
# iterations = 2000
# learning_rate = 1.0
# hidden_nodes = 10
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.078 ... Validation loss: 0.149 ... iter: 2000 ... learn: 1.0 ... hidden: 10
# iterations = 2000
# learning_rate = 1.0
# hidden_nodes = 10
# output_nodes = 1

# THIS WORKIED BUT TIMED OUT DURING SUBMISSION CHECK
# Waiting for results...Done!

# Results:
# --------
# "The process exceeded the timeout of 240 seconds."
# For help troubleshooting, please see the FAQ:
#  https://project-assistant.udacity.com/faq
# Progress: 100.0% ... Training loss: 0.057 ... Validation loss: 0.143 ... iter: 20000 ... learn: 0.15 ... hidden: 10
# iterations = 20000
# learning_rate = 0.15
# hidden_nodes = 10
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.080 ... Validation loss: 0.160
iterations = 2750
learning_rate = 0.75
hidden_nodes = 10
output_nodes = 1
