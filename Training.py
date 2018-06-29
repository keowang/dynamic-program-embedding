'''
Created on Aug 11, 2017

@author: wangke
'''

from tensorflow.contrib import rnn

import tensorflow as tf


num_epochs = 100
learning_rate = 0.0001
n_hidden = 200


vocabulary_size = 10000 # To be changed: input vocabulary  
CLASSES = 100 # To be changed: prediction classes
batch_size = 100 # To be changed
program_number = 100 # To be changed: number of programs

class Training:
    '''
    classdocs
    '''

    def __init__(self, all_program_symbol_traces, trace_lengths, labels, test_all_program_symbol_traces, test_trace_lengths, test_labels):
        '''
        Constructor
        '''
        self.all_program_symbol_traces = all_program_symbol_traces
        self.trace_lengths = trace_lengths        
        self.labels = labels                         
        
        self.test_all_program_symbol_traces = test_all_program_symbol_traces
        self.test_trace_lengths = test_trace_lengths        
        self.test_labels = test_labels                         

                
    def train_evaluate(self):
        
        # trace inputs for training: each variable trace forms one data input, meaning one program may consist of multiple traces         
        x = tf.placeholder(tf.int32, [batch_size, None]) 
        # length of each variable trace before padding
        seq_lengths = tf.placeholder(tf.int32, [batch_size]) 
        # labels for training
        y = tf.placeholder(tf.int32, [batch_size])
        
#         keep_prob = tf.constant(1.0)

        W = tf.Variable(tf.random_normal([n_hidden, CLASSES]))        
        b = tf.Variable(tf.random_normal([CLASSES]))
        
        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [vocabulary_size, n_hidden])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    
        # RNN
        cell = rnn.GRUCell(n_hidden)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
                
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seq_lengths, initial_state=init_state)


#         rnn_inputs = tf.nn.dropout(rnn_inputs, keep_prob)
#         rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
         
        # remove padding effects
        last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seq_lengths-1], axis = 1))
        
        # again one program may have multiple variable traces so split with the number of programs instead of batches/variables
        list_of_program_tensors = tf.split(last_rnn_output, program_number, 0)         
        
        all_programs_tensors = []
        for program_tensor in list_of_program_tensors:

            summed_reduced_program_tensor = tf.reduce_max(program_tensor, 0)
#             states_embedding_each_training_program = tf.reduce_mean(states_embedding_each_training_program, 0)
#             states_embedding_each_training_program = tf.reduce_sum(states_embedding_each_training_program, 0)                                
#             states_embedding_each_training_program = tf.reduce_logsumexp(states_embedding_each_training_program, 0)
            
            all_programs_tensors.append(summed_reduced_program_tensor)

        all_programs_tensors = tf.stack(all_programs_tensors, 0)
      
        prediction = tf.matmul(all_programs_tensors, W) + b                   
        correct_pred = tf.equal(tf.cast(tf.argmax(prediction,1), tf.int32), y)        
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))            
                
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))         
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
        
        with tf.Session() as sess:            
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for i in range(num_epochs):
                                      
                # Demo purpose only for one batch add a for loop to read multiple batches 
                _,_loss = sess.run(
        
                    [train_step, loss],

                    feed_dict={     
                        x : self.all_program_symbol_traces,       
                        y : self.labels,
                        seq_lengths : self.trace_lengths,
                    })

                print("training iteration is %s and total_loss: %s "%(i,_loss))                                                                                       


                
            _accuracy = sess.run(
                
                accuracy,
                
                feed_dict={     
                        x : self.test_all_program_symbol_traces,       
                        y : self.test_labels,
                        seq_lengths : self.test_trace_lengths,
                    })
                
            print("The accuracy is %s"%(_accuracy))
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
