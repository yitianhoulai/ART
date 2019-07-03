# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:03:13 2018

@author: JYANG022
"""

import numpy as np
import tensorflow as tf
import scipy.special as sp
#import math
#import matplotlib.pyplot as plt
#%matplotlib inline
slim=tf.contrib.slim
import tensorflow.contrib.distributions as ds
import DataProcess_TruthDiscovery as dataProcess
#import tensorflow_probability.python.distributions as tfd
def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), -1,keepdims=1)

tf.reset_default_graph()  
kmax=6   
n_agents=209#80
n_events=245#50#200        
n_states=4 #6
n_reliability_d1=6 #6
n_reliability_d2=6 #6
n_training=600
#M,MV_result,Ground_Truth=dataProcess.Process(n_states,'DataACT0.csv')
#AdjaciencyMatrix=dataProcess.Process_SocialNetwork(n_agents,"YObs0_cluster.csv",'From_adjaciency_matrix')
M,MV_result,Ground_Truth=dataProcess.Process(n_states,'DataACT_Real_Modify_Level4.csv')

#for validation
#M=np.array(M)[:,0:50]
#MV_result=MV_result[0:50]
#Ground_Truth=Ground_Truth[0:50]
#for validation

AdjaciencyMatrix,LaplacianMatrix=dataProcess.Process_SocialNetwork(n_agents,"YObs_Real_Filter_Edges_Modify.csv",'From_edge_list')

temperature=0.1
M_nan_or_not=tf.cast(tf.tile(tf.expand_dims(np.where(np.isnan(M)==True, 0,1),-1),[1,1,n_states]),tf.float32)
np.random.seed(1)
tf.set_random_seed(1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        #8


n_reliability=n_reliability_d1*n_reliability_d2
#m=tf.random_uniform((kmax,n_reliability_d1,n_reliability_d2), minval=-0.3, maxval=-0.35,dtype=tf.float32) 
m=np.tile(np.diag(np.ones(n_reliability_d1)),[kmax,1,1])+np.random.uniform(-0.2, -0.1,(kmax,n_reliability_d1,n_reliability_d2))
m=softmax(m)
v=0.1*np.float32(np.ones((kmax,n_reliability_d1,n_reliability_d2)))

alpha=np.float32(np.ones([n_agents,kmax]))
#alpha[0:20,0]=10*alpha[0:20,0]
#alpha[20:40,1]=10*alpha[20:40,1]
#alpha[40:60,2]=10*alpha[40:60,2]
#alpha[60:80,3]=10*alpha[60:80,3]

neural_network_size = \
    dict(
            n_theta=n_states,
            n_c=n_states,          # Number of nodes in NN decoder imput layer:n_c+n_theta change 5
            n_hidden_decoder_1=64, # Number of nodes in NN decoder Hidden layer 1 (not the input layer)
            n_hidden_decoder_2=128,  # Number of nodes in NN decoder Hidden layer 2 
            n_hidden_decoder_3=256,
            n_logits_M=n_states,        # Number of nodes in NN decoder output layer # states changed to 4. Need to change n_states to n_states+1                             
                                        
           
            n_hidden_encoder_theta_1=256, #Number of node in NN encoder Hidden layer 1 (not the input layer)
            n_hidden_encoder_theta_2=128,  #Number of node in NN encoder Hidden layer 2 
            n_hidden_encoder_theta_3=64,
            n_hidden_encoder_c_1=256, #Number of node in NN encoder Hidden layer 1 (not the input layer)
            n_hidden_encoder_c_2=128,  #Number of node in NN encoder Hidden layer 2 
            n_hidden_encoder_c_3=128,
            
            n_logits_theta=n_states, #Number of node in NN encoder output layer for theta change 6
            n_logits_c=n_reliability,   #Number of node in NN encoder output layer for c           
         )  # dimensionality of latent space
variance_p_c_given_ctilder=0.1 
variance_q_c_given_M=0.1*np.float32(np.ones([n_agents,n_reliability_d1,n_reliability_d2]))
#variance_p_c_given_eta=0.01*tf.tile(tf.expand_dims(tf.expand_dims(tf.diag(tf.ones(n_agents)),0),0),[n_reliability_d1,n_reliability_d2,1,1])


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
        #self.var_Variational_ctilder_variance=tf.placeholder(tf.float32,[kmax,n_reliability_d1,n_reliability_d2])
        #self.var_Variational_z=tf.placeholder(tf.float32,[n_agents,n_agents,kmax])
        
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
        
    def _initialize_Variables(self):
        all_variables = dict()            
                
        all_variables['tao']=tf.Variable(temperature,name="temperature")        
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            all_variables['weights_encoder_theta'] = {
                'h1': tf.get_variable('h1',[n_agents*n_states, neural_network_size['n_hidden_encoder_theta_1']]),
                'h2': tf.get_variable('h2',[neural_network_size['n_hidden_encoder_theta_1'], neural_network_size['n_hidden_encoder_theta_2']]),
                'h3': tf.get_variable('h3',[neural_network_size['n_hidden_encoder_theta_2'], neural_network_size['n_hidden_encoder_theta_3']]),
                'out_mean': tf.get_variable('out_mean',[neural_network_size['n_hidden_encoder_theta_3'], neural_network_size['n_logits_theta']])}
                #,'out_log_sigma': tf.Variable(xavier_init(n_hidden_encoder_2, n_V_theta))}
            all_variables['biases_encoder_theta'] = {
                'b1': tf.Variable(tf.zeros([neural_network_size['n_hidden_encoder_theta_1']], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([neural_network_size['n_hidden_encoder_theta_2']], dtype=tf.float32)),
                'b3': tf.Variable(tf.zeros([neural_network_size['n_hidden_encoder_theta_3']], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([neural_network_size['n_logits_theta']], dtype=tf.float32))}
                
        return all_variables
    def _create_loss_optimizer(self): 
        
        self.all_variables=self._initialize_Variables()  
       
       
        
        logits_theta=self._encoder_network_theta(self.all_variables['weights_encoder_theta'],self.all_variables['biases_encoder_theta'])
        self.q_theta = tf.nn.softmax(logits_theta)

        log_q_theta = tf.log(self.q_theta+1e-20)
       
       
        self.theta = gumbel_softmax(logits_theta,self.all_variables['tao'])
        
        
        logits_c_1=self._encoder_network_c(neural_network_size['n_hidden_encoder_c_1'], neural_network_size['n_hidden_encoder_c_2'],neural_network_size['n_hidden_encoder_c_3'], neural_network_size['n_logits_c'])
        
        logits_c=tf.reshape(logits_c_1,[-1,n_reliability_d1,n_reliability_d2])
        
        
        eps=tf.random_normal((n_agents,n_reliability_d1,n_reliability_d2),0,1,dtype=tf.float32)
        
        
        #self.c=tf.nn.softmax(logits_c)+tf.multiply(eps,variance_q_c_given_M)
        
        self.c=logits_c+tf.multiply(eps,variance_q_c_given_M)
        
        self.c_flatten=tf.reshape(self.c,[-1,n_reliability]) 
        #self.c=tf.clip_by_value(self.c,1e-20,1.0)
        #self.log_c=tf.log(self.c)
        
        #log_q_c= -tf.log(variance_q_c_given_M)#+ds.Normal(loc=0.0,scale=1.0).log_prob(eps)
        
        #log_q_c = tf.log(q_c+1e-20)
        
        
        
       
        
       
        
        
        logits_M_1 = self._decoder_network( neural_network_size['n_hidden_decoder_1'], neural_network_size['n_hidden_decoder_2'],neural_network_size['n_hidden_decoder_3'],neural_network_size['n_logits_M'])

        #logits_M_2= tf.reshape(logits_M_1,[-1,n_agents,n_states])

        #logits_M= tf.reshape(tf.nn.softmax(logits_M_1),[n_agents,n_events*n_states]) 
        logits_M= logits_M_1        
        
        
      
        #mean_p_prior_c=self._update_prior_c(self.var_Variational_s,self.var_Variational_ctilder_mean,self.var_Variational_ctilder_variance,self.var_Variational_eta_mean,self.var_Variational_eta_variance_inverse)
        mean_p_prior_c=tf.tensordot(self.var_Variational_s,self.var_Variational_ctilder_mean,[1,0])
        #p_prior_log_c=ds.Normal(loc=mean_p_prior_log_c,scale=variance_p_logc_given_ctilder)
        
        
        p_M = ds.Bernoulli(logits=logits_M)

        kl_theta_tmp = self.q_theta*(log_q_theta-tf.log(self.MV_distribution+1e-20))
        KL_theta = tf.reduce_sum(kl_theta_tmp)
        
        
        #identity_matrix=np.tile(np.diag(np.ones(n_reliability_d1)),[n_agents,1,1])
        kl_c_tmp = 0.5*tf.square(self.c-mean_p_prior_c)/variance_p_c_given_ctilder#- p_prior_log_c.log_prob(self.log_c)
        KL_c = tf.reduce_sum(kl_c_tmp)

        elbo=tf.reduce_sum(tf.multiply(p_M.log_prob(self.M_onehot),M_nan_or_not)) - KL_theta- KL_c

        self.cost=-elbo   
        
        
              
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #self.late_optimizer = \
        #    tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
    def _decoder_network(self, n_layer1, n_layer2, n_layer3, n_layer_output):
        
        theta_extent=tf.expand_dims(self.theta,0) #change12
        tensor_theta=tf.tile(theta_extent,[n_agents,1,1])
        
        #discrete_theta=tf.expand_dims(tf.argmax(self.theta,axis=1),-1)
        #tensor_c=tf.transpose(self.c,[1,0,2])
        #tensor_c=tf.transpose(tf.gather_nd(tensor_c,discrete_theta),[1,0,2]) # states changed to 4. Need to change n_states to n_states-1   
        c_extent=tf.expand_dims(self.c_flatten,1) #change12
        tensor_c=tf.tile(c_extent,[1,n_events,1])
        tensor_concat=tf.concat([tensor_theta,tensor_c],2)  #
        
        #net = slim.stack(tensor_concat,slim.fully_connected,[n_layer1,n_layer2,n_layer3])
        net = slim.fully_connected(tensor_concat, n_layer1,activation_fn=tf.nn.relu,normalizer_fn=None)
        net = slim.fully_connected(net, n_layer2,activation_fn=tf.nn.relu,normalizer_fn=None)
        net = slim.fully_connected(net, n_layer3,activation_fn=tf.nn.relu,normalizer_fn=None)
        logits_M_1 = slim.fully_connected(net,n_layer_output,activation_fn=None)        
        
        return logits_M_1
   
    def _encoder_network_theta(self, weights, biases):
        
        #weight_temp=tf.reshape(weights['h1'],[n_agents,n_states,neural_network_size['n_hidden_encoder_theta_1']])
        #weight_temp=tf.tensordot(LaplacianMatrix,weight_temp,[1,0])
        #weight_temp=tf.reshape(weight_temp,[n_agents*n_states,neural_network_size['n_hidden_encoder_theta_1']])
        
        layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(tf.cast(self.M_T_input, tf.float32),weights['h1']), 
                                           biases['b1'])) 
        dropout = tf.layers.dropout(inputs=layer_1, rate=0.5)
        
        layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(dropout, weights['h2']), 
                                           biases['b2'])) 
        layer_3 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_2, weights['h3']), 
                                           biases['b3'])) 
        logits_theta =tf.add(tf.matmul(layer_3, weights['out_mean']),
                        biases['out_mean'])
        
        return logits_theta
    
    def _encoder_network_c(self, n_layer1, n_layer2, n_layer3, n_layer_output):
        
        #net = slim.stack(self.M_input,slim.fully_connected,[n_layer1,n_layer2])
       
        net = slim.fully_connected(self.M_input, n_layer1,activation_fn=tf.nn.relu,normalizer_fn=None)
        net = slim.fully_connected(net, n_layer2,activation_fn=tf.nn.relu,normalizer_fn=None)
        net = slim.fully_connected(net, n_layer3,activation_fn=tf.nn.relu,normalizer_fn=None)
        logits_c = tf.reshape(slim.fully_connected(net,n_layer_output,activation_fn=None,normalizer_fn=None),[-1,n_layer_output])
        
    
        return logits_c

   
        
    def fit(self, M, var_Variational_s, var_Variational_ctilder_mean):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """       
        
        opt,cost,c= self.sess.run((self.optimizer, self.cost, self.c), feed_dict={self.M: M,self.var_Variational_s:var_Variational_s,self.var_Variational_ctilder_mean:var_Variational_ctilder_mean})
        #print("ctilder",self.sess.run((self.ctilder),feed_dict={self.M: M}))
        #print("Term0",self.sess.run((self.Term0),feed_dict={self.M: M}))
        return cost,c
     
def HirPara_Update(c,var_Variational_s,var_Variational_z,E_log_pi,var_Variational_ctilder_mean,var_Variational_ctilder_variance,rho):          
            
            var_Variational_ctilder_mean_out,var_Variational_ctilder_variance_out=_update_variational_parameter_of_ctilder(c,var_Variational_s,var_Variational_ctilder_mean,var_Variational_ctilder_variance,rho)
            var_Variational_z_out,E_log_pi_out=_update_variational_parameter_of_z(var_Variational_z,E_log_pi,var_Variational_s)
            var_Variational_s_out=_update_variational_parameter_of_s(c,E_log_pi_out,var_Variational_ctilder_mean_out,var_Variational_ctilder_variance_out,var_Variational_s,rho)
            
            return E_log_pi_out,var_Variational_s_out,var_Variational_ctilder_mean_out,var_Variational_ctilder_variance_out, var_Variational_z_out
           
           
def _update_variational_parameter_of_s(c,E_log_pi,var_Variational_ctilder_mean,var_Variational_ctilder_variance,var_Variational_s,rho):
                
                
            #E_log_pi=sp.digamma(var_Variational_pi)-sp.digamma(np.sum(var_Variational_pi,axis=1,keepdims=True))
            Term1=np.tile(np.expand_dims(var_Variational_ctilder_variance,0),[n_agents,1,1,1])
            
            Variational_ctilder_mean=np.tile(np.expand_dims(var_Variational_ctilder_mean,0),[n_agents,1,1,1])        
            c=np.tile(np.expand_dims(c,1),[1,kmax,1,1])        
            Term2=np.square(Variational_ctilder_mean-c)
            
            #identity_matrix=np.tile(np.diag(np.ones(n_reliability_d1)),[n_agents,kmax,1,1])
            #variance_p_c_here=tf.tile(tf.expand_dims(variance_p_logc_given_ctilder,1),[1,kmax,1,1])
            E_log_normal=-(Term1+Term2)/(2*variance_p_c_given_ctilder)        
            
            #Variational_s=Variational_s/(tf.reduce_sum(Variational_s,axis=1,keepdims=True)+1e-20) # 1e-20 is used to prevent all the elements in a row of  Variational_s are 0
            log_var_Variational_s=np.log(var_Variational_s+1e-20)
            #Variational_s=gumbel_softmax(tf.tile(E_log_pi,[n_agents,1])+tf.reduce_sum(E_log_normal,axis=[2,3]),self.all_variables['tao']) # 50 is used to prevent overflow  
            Variational_s=softmax(log_var_Variational_s+rho*(E_log_pi+np.sum(E_log_normal,axis=(2,3))-log_var_Variational_s)) # exponential 1
            #Variational_s=exp_normalize(tf.tile(E_log_pi,[n_agents,1])+tf.reduce_sum(E_log_normal,axis=[2,3]),1) # exponential 1
            return Variational_s
            
def _update_variational_parameter_of_ctilder(c,var_Variational_s,var_Variational_ctilder_mean,var_Variational_ctilder_variance,rho):
            
            variational_ctilder_mean_hat_old=np.divide(var_Variational_ctilder_mean,var_Variational_ctilder_variance)
            variational_ctilder_variance_hat_old=-0.5/var_Variational_ctilder_variance
            
            Term1=1/v
            Variational_s=np.tile(np.expand_dims(np.expand_dims(var_Variational_s,2),3),[1,1,n_reliability_d1,n_reliability_d2])
            #variance_p_c_here=tf.tile(tf.expand_dims(variance_p_logc_given_ctilder,1),[1,kmax,1,1])
            Term2=np.sum(Variational_s/variance_p_c_given_ctilder,axis=0)
            Term3=m*Term1
            Term4=np.tensordot(np.transpose(var_Variational_s),(c/variance_p_c_given_ctilder),[1,0])
            
            variational_ctilder_mean_hat=variational_ctilder_mean_hat_old+rho*(Term3+Term4-variational_ctilder_mean_hat_old)
            variational_ctilder_variance_hat=variational_ctilder_variance_hat_old+rho*(-0.5*(Term1+Term2)-variational_ctilder_variance_hat_old)
            
            #variational_ctilder_mean=(Term3+Term4)/(Term1+Term2)
            #variational_ctilder_mean=tf.clip_by_value(variational_ctilder_mean,-10,0)
            variational_ctilder_variance=-0.5/variational_ctilder_variance_hat
            variational_ctilder_mean=variational_ctilder_mean_hat*variational_ctilder_variance
            return variational_ctilder_mean, variational_ctilder_variance
def _update_variational_parameter_of_z(var_Variational_z,E_log_pi,var_Variational_s):
    AdjaciencyMatrix_tile=np.tile(np.expand_dims(AdjaciencyMatrix,-1),[1,1,kmax])
   
    var_Variational_pi_temp=np.zeros([n_agents,kmax],dtype=np.float32)
    var_Variational_pi=np.zeros([n_agents,kmax],dtype=np.float32)
    for i in range(n_agents):
        for j in range(i+1,n_agents):
            
            Term1=var_Variational_z[i,j,:]*(np.log(1.0-0.8))*(1-AdjaciencyMatrix_tile[i,j])
            Term2=var_Variational_z[i,j,:]*(np.log(0.8)-np.log(1e-10))*AdjaciencyMatrix_tile[i,j]
            Term3=E_log_pi[j,:]                
            var_Variational_z[j,i,:]=softmax(Term1+Term2+Term3)                
            var_Variational_pi_temp[j,:]=var_Variational_pi_temp[j,:]+var_Variational_z[j,i,:]
            
            Term4=var_Variational_z[j,i,:]*(np.log(1.0-0.8))*(1-AdjaciencyMatrix_tile[j,i])
            Term5=var_Variational_z[j,i,:]*(np.log(0.8)-np.log(1e-10))*AdjaciencyMatrix_tile[j,i]
            Term6=E_log_pi[i,:]                
            var_Variational_z[i,j,:]=softmax(Term4+Term5+Term6)          
            var_Variational_pi_temp[i,:]=var_Variational_pi_temp[i,:]+var_Variational_z[i,j,:]
        
        var_Variational_pi[i,:]=alpha[i,:]+var_Variational_pi_temp[i,:]+np.sum(var_Variational_s,axis=0,keepdims=1) #+var_Variational_s[i,:]
        E_log_pi[i,:]=sp.digamma(var_Variational_pi[i,:])-sp.digamma(np.sum(var_Variational_pi[i,:],keepdims=True))

            
    return var_Variational_z,E_log_pi  
def train(data, learning_rate=0.0001,training_epochs=10, display_step=1):
    
    vae = TruthDiscovery_VAE(learning_rate=learning_rate)
   
    mylog=open("./mylog.txt",'w+')
    
    var_Variational_ctilder_mean_init=m #change14    
    var_Variational_ctilder_variance_init=v
        
    var_Variational_s_init=np.float32(np.ones([n_agents,kmax])/kmax)
    var_Variational_z_init=np.float32(np.ones([n_agents,n_agents,kmax])/kmax)
    E_log_pi_init=np.log(softmax(np.random.uniform(0.9,1,[n_agents,kmax])))
    #_,c = vae.fit(data, var_Variational_s_init, var_Variational_ctilder_mean_init)
    
    #var_Variational_pi,var_Variational_s,var_Variational_ctilder_mean,var_Variational_ctilder_variance, var_Variational_z=vae._create_HirPara_Update(c)
    
    var_Variational_s=var_Variational_s_init
    var_Variational_z=var_Variational_z_init
    var_Variational_ctilder_mean=var_Variational_ctilder_mean_init
    var_Variational_ctilder_variance=var_Variational_ctilder_variance_init
    E_log_pi=E_log_pi_init
    # Training cycle
    for epoch in range(training_epochs): 
        
        rho=np.power((10+epoch),-0.6)
        # Fit training using batch data
        cost,c = vae.fit(data,var_Variational_s, var_Variational_ctilder_mean)
        
        #E_log_pi,var_Variational_s,var_Variational_ctilder_mean,var_Variational_ctilder_variance, var_Variational_z= HirPara_Update(c,var_Variational_s,var_Variational_z,E_log_pi)
               
        E_log_pi,var_Variational_s,var_Variational_ctilder_mean,var_Variational_ctilder_variance, var_Variational_z= HirPara_Update(c,var_Variational_s,var_Variational_z,E_log_pi,var_Variational_ctilder_mean,var_Variational_ctilder_variance,rho)

        
        
       
        #print("Variational_pi",vae.sess.run(vae.all_variables['Variational_pi'],feed_dict={vae.M: M}))
        #print("Variational_s",vae.sess.run((vae.all_variables['Variational_s']),feed_dict={vae.M: M}))
        #print("V_theta",vae.sess.run((vae.V_theta),feed_dict={vae.M: M}),file=mylog)
        #print("theta",vae.sess.run(tf.transpose(vae.theta),feed_dict={vae.M: M}),file=mylog)
        # Display logs per epoch step
        if epoch % display_step == 0:   
            #print("Variational_pi",vae.sess.run((vae.all_variables['weights_encoder_theta']),feed_dict={vae.M: M}))
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(cost))   #,"E_log_pi",E_log_pi_np    
            theta_output=vae.sess.run(vae.q_theta, feed_dict={vae.M: M})  
            accuracy=1-np.count_nonzero(np.argmax(theta_output,axis=1)-Ground_Truth.truths.values)/n_events  
            print("Accuracy",accuracy)
        if epoch % 100==0:
             print("c",c,file=mylog) 
             print("Variational_ctilder_mean",var_Variational_ctilder_mean,file=mylog)
             print("Variational_ctilder_variance",var_Variational_ctilder_variance,file=mylog)
          
             print("E_log_pi",E_log_pi,file=mylog) 
            
             print("Variational_s",var_Variational_s,file=mylog)       
             print("s",np.argmax(var_Variational_s,axis=1),file=mylog)  
               
             theta_output=vae.sess.run(vae.q_theta, feed_dict={vae.M: M})  
             accuracy=1-np.count_nonzero(np.argmax(theta_output,axis=1)-Ground_Truth.truths.values)/n_events  
             print("theta",theta_output,np.argmax(theta_output,axis=1),file=mylog) 
             print("Accuracy",accuracy,file=mylog)
    
    with vae.sess.as_default(): 
        
        print("softmax_c",c,file=mylog) 
        print("softmax_Variational_ctilder_mean",var_Variational_ctilder_mean,file=mylog)
        print("Variational_ctilder_variance",var_Variational_ctilder_variance,file=mylog)
        #print("Variational_ctilder_variance",vae.sess.run(vae.all_variables['Variational_ctilder_variance'], feed_dict={vae.M: M}),file=mylog)
        #print("Variational_ctilder_term1",vae.sess.run(vae.Term1, feed_dict={vae.M: M}),file=mylog) 
        #print("Variational_ctilder_term2",vae.sess.run(vae.Term2, feed_dict={vae.M: M}),file=mylog) 
        #print("Variational_ctilder_term3",vae.sess.run(vae.Term3, feed_dict={vae.M: M}),file=mylog) 
        #print("Variational_ctilder_term4",vae.sess.run(vae.Term4, feed_dict={vae.M: M}),file=mylog)  
        print("E_log_pi",E_log_pi,file=mylog) 
        #print("E_log_normal",vae.sess.run(vae.E_log_normal, feed_dict={vae.M: M}),file=mylog)                
        
        print("Variational_s",var_Variational_s,file=mylog)       
        print("s",np.argmax(var_Variational_s,axis=1),file=mylog)  
        #print("p_prior_c",vae.sess.run(vae.p_prior_c, feed_dict={vae.M: M}),file=mylog)           
        theta_output=vae.sess.run(vae.q_theta, feed_dict={vae.M: M})  
        accuracy=1-np.count_nonzero(np.argmax(theta_output,axis=1)-Ground_Truth.truths.values)/n_events  
        print("theta",theta_output,np.argmax(theta_output,axis=1),file=mylog) 
        print("Accuracy",accuracy)
               
    return vae

vae = train(data=M, training_epochs=n_training) #Problem: 20 may be too small
