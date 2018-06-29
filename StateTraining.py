'''
Created on Aug 12, 2017

@author: wangke
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

num_epochs = 100
learning_rate = 0.0001
n_hidden = 200

vocabulary_size = 10000 # To be changed: input vocabulary  
CLASSES = 100 # To be changed: prediction classes
batch_size = 10000 # To be changed
program_number = 100 # To be changed: number of programs

class StateTraining:
    '''
    classdocs
    '''

    def __init__(self, all_symbol_state_traces, variable_state_trace_lengths, variable_variable_trace_lengths, one_hot_encoding_vectors,\
                 test_all_symbol_state_traces, test_variable_state_trace_lengths, test_variable_variable_trace_lengths, test_one_hot_encoding_vectors):
        '''
        Constructor
        '''
        self.all_symbol_state_traces = all_symbol_state_traces
        self.variable_state_trace_lengths = variable_state_trace_lengths

        self.variable_variable_trace_lengths = variable_variable_trace_lengths
        self.one_hot_encoding_vectors = one_hot_encoding_vectors
        
        self.test_all_symbol_state_traces = test_all_symbol_state_traces
        self.test_variable_state_trace_lengths = test_variable_state_trace_lengths

        self.test_variable_variable_trace_lengths = test_variable_variable_trace_lengths
        self.test_one_hot_encoding_vectors = test_one_hot_encoding_vectors
        
    
    def train_evaluate(self):
        
        # one state (a tuple of variable values) from one program will be one data input  
        x = tf.placeholder(tf.int32, [batch_size, None]) 
        # the length of each data input before padding
        vv = tf.placeholder(tf.int32, [batch_size]) 
        # the length of each program (number of states) before padding.
        vs = tf.placeholder(tf.int32, [program_number]) 
        y = tf.placeholder(tf.int32, [program_number, CLASSES])
        
        W = tf.Variable(tf.random_normal([n_hidden, CLASSES]))        
        b = tf.Variable(tf.random_normal([CLASSES]))
        
        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [vocabulary_size, n_hidden])
        embedding_rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
        
        with tf.variable_scope("embedding"):
            rnn_embedding_cell = rnn.GRUCell(n_hidden)        
            all_programs_states_outputs_embedding, _ = tf.nn.dynamic_rnn(rnn_embedding_cell, embedding_rnn_inputs, sequence_length=vv, dtype=tf.float32)
            all_programs_states_embedding = tf.gather_nd(all_programs_states_outputs_embedding, tf.stack([tf.range(batch_size), vv-1], axis = 1))

        batch_of_program_state_embeddings = tf.split(all_programs_states_embedding, program_number, 0)
        batch_of_program_state_embeddings = tf.stack(batch_of_program_state_embeddings, 0)
        
        with tf.variable_scope("prediction"):
            rnn_prediction_cell = rnn.GRUCell(n_hidden)        
            rnn_prediction_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_prediction_cell] * 2)
            all_programs, _ = tf.nn.dynamic_rnn(rnn_prediction_cell, batch_of_program_state_embeddings, sequence_length=vs, dtype=tf.float32)
            all_programs_final_state = tf.gather_nd(all_programs, tf.stack([tf.range(program_number), vs-1], axis = 1))

        prediction = tf.matmul(all_programs_final_state, W) + b                   
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))        
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))            
                
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))         
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        with tf.Session() as sess:            
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for i in range(num_epochs):
                            
                # Demo purpose only for one batch add a for loop to read multiple batches 

                _,_loss = sess.run(
        
                    [train_step, loss],

                    feed_dict={     
                        x : self.all_symbol_state_traces,       
                        y : np.stack(self.one_hot_encoding_vectors, axis = 0),
                        vs : self.variable_state_trace_lengths,
                        vv : self.variable_variable_trace_lengths,
                    })

                print("training iteration is %s and total_loss: %s "%(i,_loss))                                                                                       
                

            _accuracy = sess.run(
                
                accuracy,
                
                feed_dict={     
                        x : self.test_all_symbol_state_traces,       
                        y : np.stack(self.test_one_hot_encoding_vectors, axis = 0),
                        vs : self.test_variable_state_trace_lengths,
                        vv : self.test_variable_variable_trace_lengths,                            
                    })
            
            print("The accuracy is %s"%_accuracy)

        
        

        
        
        

        
