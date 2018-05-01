# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 17:04:48 2017

@author: 
"""

import math
import numpy as np
import random
# Sert pour comparer les gradients num vs gradients backprop via la fonction grad_check
from numpy import linalg as LA

    # =========================================================================
    # ============= Part 1: Forward propagation and Cost functions ===========
    # =========================================================================

def sigmoid(x):
   sigmoid=1/float((1+math.exp(-x)))
   return sigmoid

def sigmoidGradient(z):
    g = np.vectorize(sigmoid)(z)*(1 - np.vectorize(sigmoid)(z))
    return g
	
#    size_network = [400,25,10]
     
# On sait que si il y a 3 couches 
# => On a 2 matrices theta1 et theta2

# On généralise si il y a size_network[0] couches alors 
# il y a theta1, theta2, ..., theta"size_network[0]-1" matrices 

#    length_theta1=hidden_layer_size*(input_layer_size+1)
#    length_theta1=size_network[1]*(size_network[0]+1)

# On généralise le nb d'éléments de chaque matrice 

	
# On part de size_network=[400,25,10]
    
# Pour chaque i on calcule la longueur
# Tant que i différent de size_network[-1]
    
# On fait : length_theta.append(25*(400+1))
# D'où length_theta[0] = 10025

# def init_theta(length_theta,size_network,theta,nn_params,theta_grad):
def init_theta(size_network,nn_params):
   # On initialise des listes vides pouvoir faire des append
   # Liste d'entiers
   length_theta = []
   # Liste de matrices pour theta et theta_grad
   theta = []
   theta_grad = [] 
   
   length_theta.append(size_network[1]*(size_network[0]+1))
   # Exemple, on fait : theta.append(nn_params[0:10025].reshape(25, (401)))
   theta.append(nn_params[0:length_theta[0]].reshape(size_network[1], (size_network[0]+1)))
   theta_grad.append(np.zeros((theta[0].shape)))

   # ATTENTION, la boucle ne tient pas compte du dernier indice de la boucle "len(size_network)-1"
   for  i in range(1,len(size_network)-1,+1):
       length_theta.append(len(nn_params)-length_theta[0])
       theta.append(nn_params[length_theta[i-1]:length_theta[i-1]+length_theta[i]].reshape(size_network[i+1], (size_network[i]+1)))
       theta_grad.append(np.zeros((theta[i].shape)))
    
   return (length_theta,theta)
	

# On calcule regul et z pour les couches 0 et 1 
# grâce à la même expression
	
def regul_terms(theta,size_network):
   global regul
   regul = 0
   for t, s in zip(theta,size_network):
#         Terme de régularisation 
#         regul = regul + np.sum(theta[i][:,1:(size_network[i]+1)]**2)
      regul = regul + np.sum(t[:,1:(s+1)]**2)
   return regul
	
def forward_prop(activation,theta,value_one_backprop):

   activations = []
   z_value = []
   
   iter=0
   # Ces valeurs ne servent pas dans les calculs z_value[0] et activations[0]
   z_value.append(0)
   activations.append(activation)
  
   for t in zip(theta):
                                  

      if iter == 0:
         z = np.dot(activations[iter],t[0].transpose())
		 
         z_value.append(z)
         activation = np.concatenate((value_one_backprop, np.vectorize(sigmoid)(z)),axis=1)
         activations.append(activation) 
		 
      else:	 
       # On doit obtenir : a_3 = np.vectorize(sigmoid)(z_3)
	     # Pourquoi utiliser une liste de matrice pour activations ?
         z = np.dot(activations[iter],t[0].transpose())

         z_value.append(z)
         activation = np.vectorize(sigmoid)(z)
         activations.append(activation)

      
      iter=iter+1  
   return (z_value,activations)

# Fonction de cout : va servir à calculer J en fonction des theta
def cost(nn_params,m,activations,K,y,lambda_value,size_network):

   # On transforme le vecteur long nn_params en liste de matrices theta
   theta = init_theta(size_network,nn_params)[1]
   regul = regul_terms(theta,size_network)
   
   J=0  
   # On boucle pour 0,1,2,...,m-1
   for i in range(m): 

      # CREER une fonction ForwardPropagation qui prend en entrée le nb de couches et le nb de neurones par couche ?
 
	  # h_of_Xi =  activations[len(size_network)-2][i,:]
      h_of_Xi =  activations[2][i,:]
	  
      y_i = np.zeros((K))
      # On change les valeurs des y[i]
      # Exple: if y = 5 then y_i = [0 0 0 0 1 0 0 0 0 0]
      y_i[y[i]] = 1
	  
      # On peut simplifier le code avec np.vectorize(math.log) = np.log
      J = J + np.sum(np.multiply((-1)*y_i,np.log(h_of_Xi)) - np.multiply(1 - y_i,np.log(1 - h_of_Xi))) 
      print ("J",J)      
    
   J = J/float(m)
   # Add regularization term
   J = J + ((float(lambda_value)/(float(2 * m)))*(regul))

   return J
    
# Si on ajoute network_size qui est le nombre de couches en comptant 
# la couche d'entrée et la couche de sortie puis le nb de neurones pour chaque couche
# On utilise network = 3,400,25,10 
   
    # =========================================================================
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    # =========================================================================
    # ================= Back propagation error avec matrices ==================
    # =========================================================================

# Va servir à calculer les gradients de J par rapport aux theta
# def backprop(m,nn_params,length_theta,size_network,activations,y_i_backprop):
def backprop(nn_params,m,length_theta,size_network,activations,y_i_backprop,value_one_backprop,z_value,lambda_value,X_with_one):

   # On transforme le vecteur long nn_params en liste de matrices theta
   theta = init_theta(size_network,nn_params)[1]

   # Partie initialisation de delta et theta_grad   
    
   # Boucle pour delta : 
   # - élément indice 0 = mat m x size_network[0]
   # - élément indice 1 = mat m x size_network[1]
   # - élément indice 0 = mat m x size_network[2]
   # On y met des valeurs tirées au sort aléatoirement par exemple comprises entre 0 et 1
   # delta[0] dimensions 25x400
   # delta[1] dimensions 10x25
   delta = [np.random.randn(m,nb_neurons) for nb_neurons in size_network[0:]]
   
   # Pourra être utile:
   # theta = [np.random.randn(y, x) for x, y in zip(size_network[:-1], size_network[1:])] 
   # biases = [np.random.randn(y, 1) for y in size_network[1:]]
  
   # theta_grad a les mêmes dimensions que theta, donc on prend simplement :
   theta_grad = theta
   length_theta_grad = length_theta
   theta_grad_reshape = []
                
   # ON CALCULE delta_3              
   # On initialise delta avec chaque valeur égale à delta "output" 
   # Il y a len(size_network) - 1 valeurs à initialiser
    
   # Boucle de l'indice de l'indice 3 de delta "3"
   
   for l in range(len(size_network)-1,-1,-1): 
	 
      if l==len(size_network)-1:
         delta[l] = activations[l] - y_i_backprop
         
      elif l!=0:
	  # NB: on calcule les delta  précédents et les theta_grad
         delta[l] = np.dot(delta[l+1],theta[l])*sigmoidGradient(np.concatenate((value_one_backprop, z_value[l]),axis=1))
		 
         theta_grad[l] = np.dot(delta[l+1].transpose(),activations[l])/m 
         theta_grad[l][:, 1:size_network[l]+1] = theta_grad[l][:, 1:size_network[l]+1] + float(lambda_value) / float(m) * theta[l][:, 1:size_network[l]+1]

		  
      else:  

         theta_grad[l] = np.dot(delta[l+1][:,1:].transpose(),X_with_one)/m
         theta_grad[l][:, 1:size_network[l]+1] = theta_grad[l][:, 1:size_network[l]+1] + float(lambda_value) / float(m) * theta[l][:, 1:size_network[l]+1]

    # =========================================================================
    # Part 3: Implement regularization with the cost function and gradients.
    # =========================================================================
    
   for i in range(0,len(size_network)-1,+1):
	  
      theta_grad_reshape.append(theta_grad[i].reshape((1,length_theta_grad[i]))[0])
      
      if i == 0:
         grad = theta_grad_reshape[i]
      else:
         grad = np.concatenate((grad,theta_grad_reshape[i]))
  
   return grad
   
 # REGARDER la forme des theta_param 
 
 
# On calcule les gradients numériques
# On a besoin de la fonction "cost"
# J = cost(m,activations,K,y,lambda_value,theta,size_network)
# Et de la fonction grad_check

# Par commodité de calculs les theta sont sous la forme d'un vecteur long
# On vérifie les valeurs des gradients obtenues avec grad_check
# à partir de la fonction de coût, sachant que tous les autres paramètres sont fixés

def cost_fun(theta_param,args_1):
   
   # Attention theta_param vecteur long vs theta = matrice
   # cost_value = cost(m,activations,K,y,lambda_value,theta_param,size_network)
   # grad_value = backprop(m,theta_param,length_theta,size_network,activations,y_i_backprop)
   # cost_value = cost(theta_param,m,activations,K,y,lambda_value,size_network)
   cost_value = cost(theta_param,*args_1)
   # grad_value = backprop(theta_param,*args_2)
   # return cost_value,grad_value
   return cost_value

def grad_check(f,theta_param,args_1,args_2):   

   # ATTENTION : Ajout de cette étape
   size_network = args_2[2]
   X_with_one = args_2[8]
   value_one_backprop = args_2[5]
   m = args_2[0]
   length_theta = args_2[1]
   y_i_backprop = args_2[4]
   lambda_value = args_2[7]
   
   K = args_1[2]
   y = args_1[3]
   
   theta = init_theta(size_network,theta_param)[1]
   z_value,activations = forward_prop(X_with_one,theta,value_one_backprop)
   args_2 = (m,length_theta,size_network,activations,y_i_backprop,value_one_backprop,z_value,lambda_value,X_with_one)

   grad =  backprop(theta_param,*args_2)
   
   # Les theta sont initialisés au préalable
   numgrad = np.zeros(theta_param.shape[0])
   perturb = np.zeros(theta_param.shape[0])
   e = 1e-4

   for p in range(theta_param.shape[0]):
      print ("Calcul n°:",p)
      perturb[p] = e
      
      theta_moins = init_theta(size_network,theta_param - perturb)[1]
      activations = forward_prop(X_with_one,theta_moins,value_one_backprop)[1]
      args_1 = (m,activations,K,y,lambda_value,size_network)
      loss1 = f(theta_param - perturb,args_1)
	   
      theta_plus = init_theta(size_network,theta_param + perturb)[1]
      activations = forward_prop(X_with_one,theta_plus,value_one_backprop)[1]
      args_1 = (m,activations,K,y,lambda_value,size_network)
      loss2 = f(theta_param + perturb,args_1)
	  
      print ("theta_param",theta_param)
      print ("theta_param - perturb",theta_param - perturb)
      print ("loss1",loss1)
	  
      print ("theta_param",theta_param)
      print ("theta_param + perturb",theta_param + perturb)
      print ("loss2",loss2)
	  
  
      numgrad[p] = (loss2 - loss1) / (2*e)
      # print ("numgrad[p]",numgrad[p])
      perturb[p] = 0
	  
      # if p == 10:
         # print ("numgrad",numgrad[0:10])
         # print ("grad",grad[0:10])
		 
         # diff_10 = LA.norm(numgrad[0:10]-grad[0:10])/LA.norm(numgrad[0:10]+grad[0:10])
         # print ("diff_10",diff_10)
         # return 

   # print ("numgrad",numgrad)
   #print ("NB grad",grad)
   diff = LA.norm(numgrad-grad)/LA.norm(numgrad+grad)
   
   print ("numgrad",numgrad)
   print ("grad",grad)
   print ("diff",diff)
   
   
   if diff < 1e-3:
      print ("Gradient check passed!")
   else:
      print ("Gradient check NOT passed!")
      
   return numgrad
   
# On appliquera en faisant :
#grad_check(cost_grad_fun,theta_params)
   

   

    
    
    
