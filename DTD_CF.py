# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:03:13 2018

@author: JYANG022
"""

import numpy as np
import tensorflow as tf
import scipy.special as sp
slim=tf.contrib.slim
import tensorflow.contrib.distributions as ds
import DataProcess_TruthDiscovery as dataProcess
import pandas as pd
def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), -1,keepdims=1)

def norm_to_1(logits):
    return logits / np.sum(logits, -1,keepdims=1)


tf.reset_default_graph()  
kmax=3  
n_agents=461#80
n_events=300#200        
n_states=5 #6
n_reliability_d1=6 #6
n_reliability_d2=3 #6
n_training=500

M,MV_result,Ground_Truth=dataProcess.Process(n_states,'CF.csv')
AdjaciencyMatrix,LaplacianMatrix=dataProcess.Process_SocialNetwork(n_agents,"CF_Topology.csv",'From_adjaciency_matrix')

temperature=0.01
M_nan_or_not=tf.cast(tf.tile(tf.expand_dims(np.where(np.isnan(M)==True, 0,1),-1),[1,1,n_states]),tf.float32)
np.random.seed(1)
tf.set_random_seed(1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        #8


n_reliability=n_reliability_d1*n_reliability_d2
m=np.random.uniform(size=(kmax,n_reliability_d1,n_reliability_d2), high=0.5, low=0.4) 

v=0.1*np.float32(np.ones((kmax,n_reliability_d1,n_reliability_d2)))

alpha=np.float32(np.ones([n_agents,kmax]))


neural_network_size = \
    dict(
            n_theta=n_states,
            n_c=n_states,          
            n_hidden_decoder_1=16, 
            n_hidden_decoder_2=32,  
            n_hidden_decoder_3=128,
            n_logits_M=n_states,                                     
                                        
           
            n_hidden_encoder_theta_1=128, 
            
            n_hidden_encoder_theta_3=32,
            n_hidden_encoder_c_1=128, 
            
            n_hidden_encoder_c_3=32,
            
            n_logits_theta=n_states, 
            n_logits_c=n_reliability,      
         )  
variance_p_c_given_ctilder=0.1 
variance_q_c_given_M=0.1*np.float32(np.ones([n_agents,n_reliability_d1,n_reliability_d2]))



def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  return y
class TruthDiscovery_VAE(object):  
    def __init__(self, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001):        
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate       
        
        self.var_Variational_s=tf.placeholder(tf.float32,[n_agents,kmax])
        self.var_Variational_ctilder_mean=tf.placeholder(tf.float32,[kmax,n_reliability_d1,n_reliability_d2])
       
        
        self.is_drop_out = tf.placeholder(tf.bool, None)

        
        
        # tf Graph input
        self.M = tf.placeholder(tf.int32, [n_agents, n_events]) 
        self.M_onehot=tf.one_hot(self.M, n_states)
        self.M_input=tf.reshape(self.M_onehot,[n_agents, n_events*n_states])
        
        self.M_T=tf.transpose(self.M)
        M_T_onehot=tf.one_hot(self.M_T, n_states)
        self.M_T_input=tf.reshape(M_T_onehot,[n_events, n_agents*n_states])
        self.MV_distribution=tf.squeeze(tf.reduce_sum(M_T_onehot,1,keep_dims=True)/tf.reduce_sum(M_T_onehot,[1,2],keep_dims=True))
        
  
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)        
   
    def _create_loss_optimizer(self): 
        
        all_variables = dict()
        
        all_variables['tao']=tf.Variable(temperature,name="temperature")      
       
        
        logits_theta=self._encoder_network_theta()
        self.q_theta = tf.nn.softmax(logits_theta)

        log_q_theta = tf.log(self.q_theta+1e-20)
       
       
        self.theta = gumbel_softmax(logits_theta,all_variables['tao'])
        
        
        logits_c_1=self._encoder_network_c()
        
        logits_c=tf.reshape(logits_c_1,[-1,n_reliability_d1,n_reliability_d2])
        
        
        eps=tf.random_normal((n_agents,n_reliability_d1,n_reliability_d2),0,1,dtype=tf.float32)
        
        
        
        self.c=logits_c+tf.multiply(eps,variance_q_c_given_M)
        
        self.c_flatten=tf.reshape(self.c,[-1,n_reliability]) 
       
        
        
        logits_M_1 = self._decoder_network()

       
        logits_M= logits_M_1        
        
        
        decay_theta = tf.Variable(1, trainable=False, dtype=tf.float32)

        self.decay_theta_op = decay_theta.assign(decay_theta )
        
       
        mean_p_prior_c=tf.tensordot(self.var_Variational_s,self.var_Variational_ctilder_mean,[1,0])
        
        
        
        p_M = ds.Bernoulli(logits=logits_M)

        kl_theta_tmp = self.q_theta*(log_q_theta-tf.log(self.MV_distribution+1e-20))
        KL_theta = tf.reduce_sum(kl_theta_tmp)
        
        
        
        kl_c_tmp = 0.5*(-1-tf.log(variance_q_c_given_M)+2*tf.square(self.c-mean_p_prior_c)/variance_p_c_given_ctilder+variance_q_c_given_M/variance_p_c_given_ctilder)#- p_prior_log_c.log_prob(self.log_c)
        KL_c = tf.reduce_sum(kl_c_tmp)

        elbo=tf.reduce_sum(tf.multiply(p_M.log_prob(self.M_onehot),M_nan_or_not)) - decay_theta* KL_theta- KL_c

        self.cost=-elbo   
        
        l2_loss=tf.losses.get_regularization_loss()

        self.cost=self.cost+l2_loss      
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
       
    def _decoder_network(self):
        
        # 
        theta_extent=tf.expand_dims(self.theta,0) #change12
        tensor_theta=tf.tile(theta_extent,[n_agents,1,1])
        
       
        c_extent=tf.expand_dims(self.c_flatten,1) #change12
        tensor_c=tf.tile(c_extent,[1,n_events,1])
        
        tensor_theta_c = tf.layers.dense(tensor_theta,                          
                                 activation=None,                             
                                 units=n_reliability,                                 
                                 use_bias=True,
                                 name='tensor_theta_c')

        
        tensor_concat=tf.concat([tensor_theta_c*tensor_c],2)  #
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=5e-3)
        
        net = tf.layers.dense(tensor_concat,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_decoder_1'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_dec_1')
        net = tf.layers.dense(net,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_decoder_2'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_dec_2')
       
        net = tf.layers.dense(net,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_decoder_3'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_dec_3')
        logits_M_1 = tf.layers.dense(net,                          
                                 activation=None,                             
                                 units=neural_network_size['n_logits_M'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_dec_4')    
        
        return logits_M_1
   
    def _encoder_network_theta(self):
        
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=5e-3)
        
       
        layer= tf.layers.dense(self.M_T_input,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_encoder_theta_1'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='layer_1')
        
        
        
        layer = tf.layers.dense(layer,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_encoder_theta_3'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='layer_3')
        
       
        
        logits_theta =tf.layers.dense(layer,                          
                                 activation=None,                             
                                 units=neural_network_size['n_logits_theta'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='logits_theta')
        return logits_theta
    
    def _encoder_network_c(self):
        
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=5e-3)
        net = tf.layers.dense(self.M_input,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_encoder_c_1'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_1')
       
        net = tf.layers.dense(net,                          
                                 activation=tf.nn.leaky_relu,                             
                                 units=neural_network_size['n_hidden_encoder_c_3'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_3')
        net = tf.layers.dense(net,                          
                                 activation=None,                             
                                 units=neural_network_size['n_logits_c'],
                                 kernel_regularizer=regularizer,
                                 use_bias=True,
                                 name='net_4')
       
        logits_c = tf.reshape(net,[-1,neural_network_size['n_logits_c']])
        
    
        return logits_c

   
        
    def fit(self, M, var_Variational_s, var_Variational_ctilder_mean,is_drop_out):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """       
        
        opt,cost,c,_= self.sess.run((self.optimizer, self.cost, self.c, self.decay_theta_op), feed_dict={self.M: M,self.var_Variational_s:var_Variational_s,self.var_Variational_ctilder_mean:var_Variational_ctilder_mean,self.is_drop_out:is_drop_out})
        
       
        return cost,c
     
def HirPara_Update(c,var_Variational_s,var_Variational_z,E_log_pi,mu_hat_old,two_sigma_hat_old,Variational_s_old,init_flag):          
    
    var_Variational_ctilder_mean_out,var_Variational_ctilder_variance_out,mu_hat_old,two_sigma_hat_old=_update_variational_parameter_of_ctilder(c,var_Variational_s,mu_hat_old,two_sigma_hat_old,init_flag)
    var_Variational_z_out,E_log_pi_out=_update_variational_parameter_of_z(var_Variational_z,E_log_pi,var_Variational_s)
    var_Variational_s_out,Variational_s_old=_update_variational_parameter_of_s(c,E_log_pi_out,var_Variational_ctilder_mean_out,var_Variational_ctilder_variance_out,Variational_s_old)
    
    return E_log_pi_out,var_Variational_s_out,var_Variational_ctilder_mean_out,var_Variational_ctilder_variance_out, var_Variational_z_out,mu_hat_old,two_sigma_hat_old,Variational_s_old
   
   
def _update_variational_parameter_of_s(c,E_log_pi,var_Variational_ctilder_mean,var_Variational_ctilder_variance,Variational_s_old):
        
        
    
    Term1=np.tile(np.expand_dims(var_Variational_ctilder_variance,0),[n_agents,1,1,1])
    
    Variational_ctilder_mean=np.tile(np.expand_dims(var_Variational_ctilder_mean,0),[n_agents,1,1,1])        
    c=np.tile(np.expand_dims(c,1),[1,kmax,1,1])        
    Term2=np.square(Variational_ctilder_mean-c)
    
    
    E_log_normal=(Term1+Term2)/(2*variance_p_c_given_ctilder)        
    
    rho=0.1
    
    Variational_s=np.power(Variational_s_old,1-rho)*np.exp(rho*(E_log_pi-np.sum(E_log_normal,axis=(2,3))))
    Variational_s=norm_to_1(Variational_s) # exponential 1
    Variational_s_old=Variational_s
    return Variational_s,Variational_s_old
    
def _update_variational_parameter_of_ctilder(c,var_Variational_s,mu_hat_old,two_sigma_hat_old,init_flag):
    Term1=1/v
    Variational_s=np.tile(np.expand_dims(np.expand_dims(var_Variational_s,2),3),[1,1,n_reliability_d1,n_reliability_d2])
    Term2=np.sum(Variational_s/variance_p_c_given_ctilder,axis=0)
    Term3=m*Term1
    Term4=np.tensordot(np.transpose(var_Variational_s),(c/variance_p_c_given_ctilder),[1,0])
    if init_flag==True:
        mu_hat=Term3+Term4
        two_sigma_hat=-(Term1+Term2)
    else:
        mu_hat=mu_hat_old-0.1*(mu_hat_old-(Term3+Term4))
        two_sigma_hat=two_sigma_hat_old-0.1*(two_sigma_hat_old+(Term1+Term2))
    
    
    variational_ctilder_variance=-1/two_sigma_hat#1/(Term1+Term2)
    variational_ctilder_mean= variational_ctilder_variance*mu_hat#(Term3+Term4)/(Term1+Term2)
    mu_hat_old=mu_hat
    two_sigma_hat_old=two_sigma_hat
    return variational_ctilder_mean, variational_ctilder_variance,mu_hat_old,two_sigma_hat_old
def _update_variational_parameter_of_z(var_Variational_z,E_log_pi,var_Variational_s):
    AdjaciencyMatrix_tile=np.tile(np.expand_dims(AdjaciencyMatrix,-1),[1,1,kmax])
   
    
    var_Variational_pi=np.zeros([n_agents,kmax],dtype=np.float32)
    
    
    Term1=var_Variational_z*(np.log(1.0-0.8))*(1-AdjaciencyMatrix_tile)
    Term2=var_Variational_z*(np.log(0.8)-np.log(1e-10))*AdjaciencyMatrix_tile
    Term3=np.tile(np.expand_dims(E_log_pi,0),[n_agents,1,1])
    var_Variational_z=np.transpose(softmax(Term1+Term2+Term3),[1,0,2])
    var_Variational_pi_temp=np.sum(var_Variational_z,axis=1,keepdims=False)
    
    
    var_Variational_pi=alpha+var_Variational_pi_temp+np.tile(np.sum(var_Variational_s,axis=0,keepdims=1),[n_agents,1])#+var_Variational_s[i,:]
    E_log_pi=sp.digamma(var_Variational_pi)-sp.digamma(np.tile(np.sum(var_Variational_pi,axis=1,keepdims=True),[1,kmax]))
    
    
            
    return var_Variational_z,E_log_pi  
def train(data, learning_rate=0.001,training_epochs=10, display_step=1):
    
    vae = TruthDiscovery_VAE(learning_rate=learning_rate)
   
    mylog=open("./mylog.txt",'w+')
    
    var_Variational_ctilder_mean_init=m #change14    
        
    var_Variational_s_init=np.float32(np.ones([n_agents,kmax])/kmax)
    var_Variational_z_init=np.float32(np.ones([n_agents,n_agents,kmax])/kmax)
    E_log_pi_init=np.log(softmax(np.random.uniform(0.9,1,[n_agents,kmax])))
    
    
    var_Variational_s=var_Variational_s_init
    var_Variational_z=var_Variational_z_init
    var_Variational_ctilder_mean=var_Variational_ctilder_mean_init
    mu_hat_old=0
    two_sigma_hat_old=0
    Variational_s_old=1
    E_log_pi=E_log_pi_init
    # Training cycle
    
    cost_val = []

    for epoch in range(training_epochs): 
        
        # Fit training using batch data
        init_flag=False
        is_drop_out=False
        if epoch==0:
            init_flag=True
        cost,c = vae.fit(data,var_Variational_s, var_Variational_ctilder_mean,is_drop_out)
        
        E_log_pi,\
        var_Variational_s,\
        var_Variational_ctilder_mean,\
        var_Variational_ctilder_variance, \
        var_Variational_z,mu_hat_old,\
        two_sigma_hat_old, \
        Variational_s_old \
        = \
        HirPara_Update(c,
        var_Variational_s,
        var_Variational_z,
        E_log_pi,
        mu_hat_old,
        two_sigma_hat_old,Variational_s_old,init_flag)
               
        
        
        cost_val.append(cost)   
      
    
        if epoch > 350 and cost_val[-1] > np.mean(cost_val[-10:-1]):  
            print("Early stopping...")
            break

        
       
        if epoch % display_step == 0:   
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(cost))       
            theta_output=vae.sess.run(vae.q_theta, feed_dict={vae.M: M,vae.is_drop_out:is_drop_out})  
            accuracy=1-np.count_nonzero(np.argmax(theta_output,axis=1)-Ground_Truth.truths.values)/n_events  
            print("Accuracy",accuracy)
      
    return vae

import time
start = time.time()
vae = train(data=M, training_epochs=n_training)
end = time.time()
print(end-start)
