import numpy as np 
import pandas as pd 
from scipy import optimize

data = pd.read_csv('trainT.csv')

x = data[['Age','Sex_female','Sex_male','Pclass_1','Pclass_2','Pclass_3']]
y = data['Survived']

def train_test_split(x,y,test_size=.3):
	index = np.random.rand(len(x))<= test_size
	x_train = np.array(x.iloc[~index])
	x_test = np.array(x.iloc[index])
	y_train = np.array(y.iloc[~index])
	y_test = np.array(y.iloc[index])
	return x_train,x_test,y_train,y_test

	



class Neural_Network:
	def __init__(self):
		#Hyperprameters
		self.epoch = 100000
		self.lr = 0.1
		self.input_layer_neurons = x.shape[1]
		self.first_hidden_layer_neurons = 6
		self.second_hidden_layer_neurons = 3
		self.output_layer_neurons = 1
		#parameters initialization
		self.w_H1 = np.random.uniform(size=(self.input_layer_neurons,self.first_hidden_layer_neurons))
		self.b_H1 = np.random.uniform(size=(1,self.first_hidden_layer_neurons))

		self.w_H2 = np.random.uniform(size=(self.first_hidden_layer_neurons,self.second_hidden_layer_neurons))
		self.b_H2 = np.random.uniform(size=(1,self.second_hidden_layer_neurons))

		self.w_OUT = np.random.uniform(size=(self.second_hidden_layer_neurons,self.output_layer_neurons))
		self.b_OUT = np.random.uniform(size=(1,self.output_layer_neurons))

	def forward_prop(self,x):
		self.first_hidden_layer_input = np.dot(x,self.w_H1)+self.b_H1
		self.first_hidden_layer_activation = self.sigmoid(self.first_hidden_layer_input)

		self.second_hidden_layer_input = np.dot(self.first_hidden_layer_activation,self.w_H2)+self.b_H2
		self.second_hidden_layer_activation = self.sigmoid(self.second_hidden_layer_input)

		self.output_layer_input = np.dot(self.second_hidden_layer_activation,self.w_OUT)+self.b_OUT
		output = self.sigmoid(self.output_layer_input)

		return output

	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def derivative_sigmoid(self,x):
		return x*(1-x)

	def cost_function(self,x,y):
		yhat = self.forward_prop(x)
		y = y.reshape((y.size,1))
		Cost = 0.5*np.sum((y - yhat)**2)
		return Cost

	def back_prop(self,x,y):
		output = self.forward_prop(x)
		self.slope_output_layer = self.derivative_sigmoid(output)
		self.slope_second_hidden_layer = self.derivative_sigmoid(self.second_hidden_layer_activation)
		self.slope_first_hidden_layer = self.derivative_sigmoid(self.first_hidden_layer_activation)

		y = y.reshape((y.size,1))
		self.E = y-output
		delta_output = self.E * self.slope_output_layer
	
		
		self.Error_at_second_hidden_layer = delta_output.dot(self.w_OUT.T)
		delta_second_hidden_layer = self.Error_at_second_hidden_layer * self.slope_second_hidden_layer

		self.Error_at_first_hidden_layer = delta_second_hidden_layer.dot(self.w_H2.T)
		delta_first_hidden_layer = self.Error_at_first_hidden_layer * self.slope_first_hidden_layer

		return delta_first_hidden_layer,delta_second_hidden_layer,delta_output

	def get_params(self):
		params = np.concatenate((self.w_H1.ravel(),self.w_H2.ravel(),self.w_OUT.ravel()))
		return params

	def set_params(self,params):
		w_H1_start = 0 
		w_H1_end = self.input_layer_neurons*self.first_hidden_layer_neurons 
		self.w_H1 = np.reshape(params[w_H1_start:w_H1_end],(self.input_layer_neurons,self.first_hidden_layer_neurons ))

		w_H2_end = w_H1_end+self.first_hidden_layer_neurons * self.second_hidden_layer_neurons
		self.w_H2 = np.reshape(params[w_H1_end:w_H2_end],(self.first_hidden_layer_neurons, self.second_hidden_layer_neurons))

		w_OUT_end = w_H2_end+self.second_hidden_layer_neurons * self.output_layer_neurons
		self.w_OUT = np.reshape(params[w_H2_end:w_OUT_end],(self.second_hidden_layer_neurons ,self.output_layer_neurons))

	def compute_grad(self,x,y):
		delta_first_hidden_layer,delta_second_hidden_layer,delta_output = self.back_prop(x,y)
		return np.concatenate((delta_first_hidden_layer.ravel(),delta_second_hidden_layer.ravel(),delta_output.ravel()))

class trainer(object):
	def __init__(self,N):
		self.N = N

	def cost_function_wrapper(self,params,x,y):
		self.N.set_params(params)
		cost = self.N.cost_function(x,y)
		grad = self.N.compute_grad(x,y)
		return cost ,grad

	def callbackF(self,params):
		self.N.set_params(params)
		self.J.append(self.N.cost_function(self.x,self.y))

	def train(self,x,y):
		self.x = x
		self.y = y

		self.J = []

		params0 = self.N.get_params()

		options = {'maxiter':200,'disp':True}
		_res = optimize.minimize(self.cost_function_wrapper, params0, jac=True, method='BFGS', args=(x,y), options=options, callback=self.callbackF)

		self.N.set_params(_res.x)
		self.optimizationResult = _res




x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.4)

NN = Neural_Network()

T  = trainer(NN)

T.train(x_train,y_train)



''''
		for i in range(epoch):
			output = self.forward_prop(x)
			

			w_OUT += second_hidden_layer_activation.T.dot(delta_output) * lr
			b_OUT += np.sum(delta_output,axis=0,keepdims=True)*lr

			w_H2 += first_hidden_layer_activation.T.dot(delta_second_hidden_layer) * lr
			b_H2 += np.sum(delta_second_hidden_layer,axis=0,keepdims=True)*lr

			w_H1 += x.T.dot(delta_first_hidden_layer) * lr
			b_H1 += np.sum(delta_first_hidden_layer,axis=0,keepdims=True) *lr

'''

