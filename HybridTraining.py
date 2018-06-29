'''
Created on Aug 15, 2017

@author: v-wangke
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

number_of_variables_lstms = 20 # To be changed: the maximum number of variables a program has among all programs 
max_trace_length = 1000 # To be changed: the maximum length of a program trace among all programs

class HybridTraining:
    '''
    classdocs
    '''

    def __init__(self, all_program_input_masks, all_program_symbol_traces, all_program_dependency_masks, one_hot_encoding_vectors, \
                 test_all_program_input_masks, test_all_program_symbol_traces, test_all_program_dependency_masks, test_one_hot_encoding_vectors):
        '''
        Constructor
        '''
        self.all_program_input_masks = all_program_input_masks
        self.all_program_symbol_traces = all_program_symbol_traces

        self.all_program_dependency_masks = all_program_dependency_masks
        self.one_hot_encoding_vectors = one_hot_encoding_vectors       

        self.test_all_program_input_masks = test_all_program_input_masks
        self.test_all_program_symbol_traces = test_all_program_symbol_traces

        self.test_all_program_dependency_masks = test_all_program_dependency_masks
        self.test_one_hot_encoding_vectors = test_one_hot_encoding_vectors       
    
    
    def retrieve_previous_hidden_states(self, list_of_all_program_h, list_of_all_program_d, i):
        
        all_programs_previous_state = []
        
        for one_program_h, one_program_d in zip(list_of_all_program_h, list_of_all_program_d):
                          
            list_of_dependencies = tf.unstack(one_program_d, axis = 0)
            one_program_current_dependency = list_of_dependencies[i]
            one_program_current_dependency_2D = tf.reshape(one_program_current_dependency, [number_of_variables_lstms, 1])
            one_program_current_dependency_float = tf.cast(one_program_current_dependency_2D, tf.float32)
            
            dependency_all = one_program_h * one_program_current_dependency_float
            previous_state = tf.reduce_prod(dependency_all, axis = 0)

            all_programs_previous_state.append(previous_state)
            
        return tf.convert_to_tensor(all_programs_previous_state)
                
    
    def update_current_hidden_states(self, list_of_all_rnn_current_state, list_of_all_program_mi, ma, list_of_all_program_h, i):
        
        list_of_all_rnn_current_state_tensor = tf.convert_to_tensor(list_of_all_rnn_current_state)
        
        list_of_all_program = tf.unstack(list_of_all_rnn_current_state_tensor, axis = 1)
        list_of_all_program_tensor = tf.convert_to_tensor(list_of_all_program)
                
        list_of_all_program_hidden_state_mask = []                        
        for one_program_mi in list_of_all_program_mi:
            
            list_of_token_input_mi_from_one_program = tf.unstack(one_program_mi, axis = 0)
            current_input_mi = list_of_token_input_mi_from_one_program[i]
            
            current_input_mi_2D = tf.reshape(current_input_mi, [number_of_variables_lstms, 1])
            current_input_mi_2D_float = tf.cast(current_input_mi_2D, tf.float32)
            
            list_of_all_program_hidden_state_mask.append(current_input_mi_2D_float)            
        list_of_all_program_hidden_state_mask_tensor = tf.convert_to_tensor(list_of_all_program_hidden_state_mask)
        
        list_of_all_program_current_state = list_of_all_program_hidden_state_mask_tensor * list_of_all_program_tensor
        
        list_of_all_program_h_tensor = tf.convert_to_tensor(list_of_all_program_h)
        
        ma_float32 = tf.cast(ma, tf.float32)        
        all_program_current_state = (ma_float32 - list_of_all_program_hidden_state_mask_tensor) * list_of_all_program_h_tensor + list_of_all_program_current_state
        
        return all_program_current_state
            
    
    def train_evaluate(self):
        
        # each program now will be represented as one single trace
        x = tf.placeholder(tf.int32, [batch_size, max_trace_length])
        
        # the tensor for keeping the m_hidden states for all variables/lstms
        hm = tf.placeholder(tf.float32, [batch_size, number_of_variables_lstms, n_hidden])        
        # the tensor for keeping the c_hidden states for all variables/lstms
        hc = tf.placeholder(tf.float32, [batch_size, number_of_variables_lstms, n_hidden]) 
        
        # mask for each input trace: indication of which lstms is responsible for each input token 
        mi = tf.placeholder(tf.int32, [batch_size, max_trace_length, number_of_variables_lstms]) 

        # complement mask to mi   
        ma = tf.constant(1, tf.int32, [batch_size, number_of_variables_lstms, 1])
        
        # dependency mask indicating dependency variables for each value in a trace
        d = tf.placeholder(tf.int32, [batch_size, max_trace_length, number_of_variables_lstms])
        
        W = tf.Variable(tf.random_normal([n_hidden * 2, CLASSES]))        
        b = tf.Variable(tf.random_normal([CLASSES]))
        y = tf.placeholder(tf.int32, [batch_size, CLASSES])
        
        # creating lstms for each of the variable 
        for i in range(number_of_variables_lstms):            
            with tf.variable_scope('variable_lstm_%s'%i):
                rnn_cell = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
                rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * 2)

        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [vocabulary_size, n_hidden])
        embedding_x = tf.nn.embedding_lookup(embeddings, x)
                
        list_of_trace_tokens_for_all_program_embedding_x = tf.unstack(embedding_x, axis = 1)
        list_of_all_program_mi = tf.unstack(mi, axis = 0)
        list_of_all_program_d = tf.unstack(d, axis = 0)
        list_of_all_program_hc = tf.unstack(hc, axis = 0)
        list_of_all_program_hm = tf.unstack(hm, axis = 0)
                
              
        for i, one_trace_token_for_all_program_embedding_x in enumerate(list_of_trace_tokens_for_all_program_embedding_x):
            print("i=",i)

            list_of_all_rnn_current_c_state = []
            list_of_all_rnn_current_m_state = []

            # calculate previous state according to dependency mask 
            all_program_previous_c_states_tensor = self.retrieve_previous_hidden_states(list_of_all_program_hc, list_of_all_program_d, i)            
            all_program_previous_m_states_tensor = self.retrieve_previous_hidden_states(list_of_all_program_hm, list_of_all_program_d, i)
            all_program_previous_states_tensor = (all_program_previous_c_states_tensor, all_program_previous_m_states_tensor)
            
            # compute next state 
            for k in range(number_of_variables_lstms):            
                with tf.variable_scope('variable_lstm_%s'%k):
                    _, all_program_current_state_tensor = rnn_cell(one_trace_token_for_all_program_embedding_x, all_program_previous_states_tensor)
                    list_of_all_rnn_current_c_state.append(all_program_current_state_tensor[0])
                    list_of_all_rnn_current_m_state.append(all_program_current_state_tensor[1])                    
                     
           
            all_program_current_c_state = self.update_current_hidden_states(list_of_all_rnn_current_c_state, list_of_all_program_mi, ma, list_of_all_program_hc, i)
            all_program_current_m_state = self.update_current_hidden_states(list_of_all_rnn_current_m_state, list_of_all_program_mi, ma, list_of_all_program_hm, i)
            
            list_of_all_program_hc = tf.unstack(all_program_current_c_state, axis = 0)
            list_of_all_program_hm = tf.unstack(all_program_current_m_state, axis = 0)
                            
        final_c_states = tf.reduce_sum(tf.convert_to_tensor(list_of_all_program_hc),axis = 1)
        final_m_states = tf.reduce_sum(tf.convert_to_tensor(list_of_all_program_hm),axis = 1)
        final_states = tf.concat([final_c_states, final_m_states], axis = 1)
        
        prediction = tf.matmul(final_states, W) + b                   
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))        
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))            
                
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))         
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

    
        with tf.Session() as sess:            
            init = tf.global_variables_initializer()
            sess.run(init)
            
            
            for i in range(num_epochs):
            
                initial_state =  np.zeros([batch_size, number_of_variables_lstms, n_hidden])
                
                # Demo purpose only for one batch add a for loop to read multiple batches 

                _,_loss = sess.run(
        
                    [train_step, loss],

                    feed_dict={     
                        x : self.all_program_symbol_traces,       
                        mi : self.all_program_input_masks,
                        d : self.all_program_dependency_masks,
                        y : self.one_hot_encoding_vectors,
                        hc : initial_state,
                        hm : initial_state,
                    })

                print("training iteration is %s and total_loss: %s "%(i,_loss))                                                                                       

            initial_state =  np.zeros([batch_size, number_of_variables_lstms, n_hidden])
            
            _accuracy = sess.run(
                
                accuracy,
                
                feed_dict={     
                        x : self.test_all_program_symbol_traces,       
                        mi : self.test_all_program_input_masks,
                        d : self.test_all_program_dependency_masks,
                        y : self.test_one_hot_encoding_vectors,
                        hc : initial_state,
                        hm : initial_state,
                    })
            
            print("The accuracy is %s"%_accuracy)
        
        
        
        

        