 
 # by R. Chandra
#https://github.com/rohitash-chandra/Bayesianlogisticreg_multioutputs

import numpy as np
import random
import math

import matplotlib.pyplot as plt

from math import exp
 

class logistic_regression:

	def __init__(self, num_epocs, train_data, test_data, num_features, learn_rate, activation):
		self.train_data = train_data
		self.test_data = test_data 
		self.num_features = num_features
		self.num_outputs = self.train_data.shape[1] - num_features 
		self.num_train = self.train_data.shape[0]
 
		#self.w = np.random.uniform(-0.5, 0.5, num_features)  # in case one output class
		self.w = np.random.uniform(-.5, .5, (num_features, self.num_outputs))  
		self.b = np.random.uniform(-.5, .5, self.num_outputs) 
		self.learn_rate = learn_rate
		self.max_epoch = num_epocs
		self.use_sigmoid = activation #SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear 
		self.out_delta = np.zeros(self.num_outputs)

		print(self.w, ' self.w init') 
		print(self.b, ' self.b init')   


 
	def activation_func(self,z_vec):
		if self.use_sigmoid == True:
			y =  1 / (1 + np.exp(z_vec)) # sigmoid/logistic 
		else:  
			y =   z_vec
		return y

	def squared_error(self, prediction, actual):
		return  np.sum(np.square(prediction - actual))/prediction.shape[0]# to cater more in one output/class
 

	def predict(self, x_vec ): # implementation using dot product
		z_vec = x_vec.dot(self.w) - self.b 
		output = self.activation_func(z_vec) # Output 
		return output

	def gradient(self, x_vec, output, actual):   
		if self.use_sigmoid == True :
			out_delta =   (output - actual)*(output*(1-output)) 
		else: # for linear activation
			out_delta =   output - actual
		return out_delta 

	def update(self, x_vec, output, actual): # implementation using for loops 
		for x in range(0, self.num_features):
			for y in range(0, self.num_outputs):
				self.w[x,y] += self.learn_rate * self.out_delta[y] * x_vec[x] 
		for y in range(0, self.num_outputs):
			self.b += -1 * self.learn_rate * self.out_delta[y]
	 

	def test_model(self, data, tolerance):  

		num_instances = data.shape[0]

		class_perf = 0
		sum_sqer = 0   
		for s in range(0, num_instances):		
			input_instance  =  self.train_data[s,0:self.num_features] 
			actual  = self.train_data[s,self.num_features:]  
			prediction = self.predict(input_instance) 
			pred_binary = np.zeros(prediction.shape[0])
			sum_sqer += self.squared_error(prediction, actual)
			index= np.argmax(prediction)
			pred_binary[index] = 1 # i=for softmax  
			#pred_binary = np.where(prediction > (1 - tolerance), 1, 0) # for sigmoid in case of classification

			if( (actual==pred_binary).all()):
				class_perf =  class_perf +1   

		rmse = np.sqrt(sum_sqer/num_instances)
		percentage_correct = float(class_perf/num_instances) * 100 
		print(percentage_correct, rmse,  ' class_perf, rmse')  

		return ( rmse, percentage_correct)

 
	def SGD(self):   
		epoch = 0 
		shuffle = True

		while  epoch < self.max_epoch:
			sum_sqer = 0
			for s in range(0, self.num_train): 
				if shuffle ==True:
					i = random.randint(0, self.num_train-1) 
				input_instance  =  self.train_data[i,0:self.num_features]  
				actual  = self.train_data[i,self.num_features:]  
				prediction = self.predict(input_instance) 
				sum_sqer += self.squared_error(prediction, actual)
				self.out_delta = self.gradient( input_instance, prediction, actual)    # major difference when compared to GD 
				self.update(input_instance, prediction, actual) 

			epoch=epoch+1  

		rmse_train, train_perc = self.test_model(self.train_data, 0.3) 
		rmse_test =0
		test_perc =0
		rmse_test, test_perc = self.test_model(self.test_data, 0.3)

		return (train_perc, test_perc, rmse_train, rmse_test) 
				
#------------ new functions added for MCMC below
	 
	def encode(self, w): # get  the parameters and encode into the model
		w_size = self.num_features * self.num_outputs
		w_temp= w[0:w_size]
		self.w = np.reshape(w_temp, (self.num_features, self.num_outputs))
		self.b = w[w_size:w.shape[0]]  

	def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

		self.encode(w)  # method to encode w and b
		fx = np.zeros((data.shape[0],self.num_outputs))
	 
		for s in range(0, data.shape[0]):  
				i = s #random.randint(0, data.shape[0]-1)  (we dont shuffle in this implementation)
				input_instance  =  data[i,0:self.num_features]  
				actual  = data[i,self.num_features:]  
				prediction = self.predict(input_instance)  
				fx[s,:] = prediction 

		return fx


	 

#------------------------------------------------------------------


class MCMC:
	def __init__(self, samples, traindata, testdata, topology, regression):
		self.samples = samples  # NN topology [input, hidden, output]
		self.topology = topology  # max epocs
		self.traindata = traindata  #
		self.testdata = testdata
		random.seed() 
		self.regression = regression # False means classification


	def rmse(self, predictions, targets):
		return np.sqrt(((predictions - targets) ** 2).mean())

	def likelihood_func(self, model, data, w, tausq):
		y = data[:, self.topology[0]:]
		fx = model.evaluate_proposal(data, w) 
		accuracy = self.rmse(fx, y) #RMSE 
		loss = np.sum(-0.5*np.log(2*math.pi*tausq) - 0.5*np.square(y-fx)/tausq)
		 
		return [loss, fx, accuracy]

	
	def likelihood_func_(self, model, data, w, tausq):
		y = data[:, self.topology[0]:]
		fx = model.evaluate_proposal(data, w) 
		accuracy = self.rmse(fx, y) #RMSE 
		n = (y.shape[0] * y.shape[1]) # number of samples x number of outputs (prediction horizon)
		p = - (n/2) * np.log(2 * math.pi * tausq) 
		log_lhood =  p - ((1/2*tausq)  *  np.sum(np.square(y- fx)) )

		return [log_lhood, fx, accuracy]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq): 
		param = (self.topology[0]  * self.topology[1]) + self.topology[1] # number of parameters in model
		part1 = -1 * (param / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
		log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss

	def sampler(self):

		# ------------------- initialize MCMC
		testsize = self.testdata.shape[0]
		trainsize = self.traindata.shape[0]
		samples = self.samples

		x_test = np.linspace(0, 1, num=testsize)
		x_train = np.linspace(0, 1, num=trainsize)

		#self.topology  # [input,   output]
		y_test = self.testdata[:, self.topology[0]:]
		y_train = self.traindata[:, self.topology[0]:]
	  
		w_size = (self.topology[0] *   self.topology[1]) +  self.topology[1]  # num of weights and bias  

		pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
		pos_tau = np.ones((samples, 1))


		fxtrain_samples = np.ones((samples, trainsize, self.topology[1]))  # fx of train data over all samples
		fxtest_samples = np.ones((samples, testsize, self.topology[1]))  # fx of test data over all samples
		rmse_train = np.zeros(samples)
		rmse_test = np.zeros(samples)

		w = np.random.randn(w_size)

		w_proposal = np.random.randn(w_size)



		step_w = 0.02;  # defines how much variation you need in changes to w
		step_eta = 0.01;  
		# eta is an additional parameter to cater for noise in predictions (Gaussian likelihood). 
		# note eta is used as tau in the sampler to consider log scale. 
		# eta is not used in multinomial likelihood. 
 

		model = logistic_regression(0 ,  self.traindata, self.testdata, self.topology[0], 0.1, self.regression) 

		pred_train = model.evaluate_proposal(self.traindata, w) 
		pred_test = model.evaluate_proposal(self.testdata, w)

		eta = np.log(np.var(pred_train - y_train)) # this is to estimate var of eta that represents noise
		tau_pro = np.exp(eta)

		print(tau_pro, 'tau_pro ')

		sigma_squared = 5  # considered by looking at distribution of  similar trained  models - i.e distribution of weights and bias
		nu_1 = 0
		nu_2 = 0

		prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

		[likelihood, pred_train, rmsetrain] = self.likelihood_func(model, self.traindata, w, tau_pro)

		print(likelihood, ' initial likelihood')
		[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(model, self.testdata, w, tau_pro)


		naccept = 0  

		for i in range(samples - 1):

			w_proposal = w + np.random.normal(0, step_w, w_size)

			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = math.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(model, self.traindata, w_proposal, tau_pro)
			[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(model, self.testdata, w_proposal, tau_pro)

			# likelihood_ignore  refers to parameter that will not be used in the alg.

			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro)  # takes care of the gradients

			diff_likelihood = likelihood_proposal - likelihood # since we using log scale: based on https://www.rapidtables.com/math/algebra/Logarithm.html
			diff_priorliklihood = prior_prop - prior_likelihood

			mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

			u = random.uniform(0, 1)

			if u < mh_prob:
				# Update position
				#print    (i, ' is accepted sample')
				naccept += 1
				likelihood = likelihood_proposal
				prior_likelihood = prior_prop
				w = w_proposal
				eta = eta_pro
				rmse_train[i + 1,] = rmsetrain
				rmse_test[i + 1,] = rmsetest


				print (i, likelihood, prior_likelihood, rmsetrain, rmsetest,   'accepted')

				pos_w[i + 1,] = w_proposal
				pos_tau[i + 1,] = tau_pro
				fxtrain_samples[i + 1,] = pred_train
				fxtest_samples[i + 1,] = pred_test 

			else:
				pos_w[i + 1,] = pos_w[i,]
				pos_tau[i + 1,] = pos_tau[i,]
				fxtrain_samples[i + 1,] = fxtrain_samples[i,]
				fxtest_samples[i + 1,] = fxtest_samples[i,] 
				rmse_train[i + 1,] = rmse_train[i,]
				rmse_test[i + 1,] = rmse_test[i,]

		 
		accept_ratio = naccept / (samples * 1.0) * 100


		print(accept_ratio, '% was accepted')

		burnin = 0.25 * samples  # use post burn in samples

		pos_w = pos_w[int(burnin):, ]
		pos_tau = pos_tau[int(burnin):, ] 
		rmse_train = rmse_train[int(burnin):]
		rmse_test = rmse_test[int(burnin):] 


		rmse_tr = np.mean(rmse_train)
		rmsetr_std = np.std(rmse_train)
		rmse_tes = np.mean(rmse_test)
		rmsetest_std = np.std(rmse_test)
		print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, ' rmse_tr, rmsetr_std, rmse_tes, rmsetest_std')


		# let us next test the Bayesian model using the posterior distributions over n trials


		num_trials = 10

		accuracy = np.zeros(num_trials)

		for i in range(num_trials):
			#print(pos_w.mean(axis=0), pos_w.std(axis=0), ' pos w mean, pos w std')
			w_drawn = np.random.normal(pos_w.mean(axis=0), pos_w.std(axis=0), w_size)
			tausq_drawn = np.random.normal(pos_tau.mean(), pos_tau.std()) # a buf is present here - gives negative values at times

			[loss, fx_,  accuracy[i]] = self.likelihood_func(model, self.testdata, w_drawn, tausq_drawn)

			print(i, loss,  accuracy[i],  tausq_drawn , pos_tau.mean(), pos_tau.std(), ' posterior test ')

		print(accuracy.mean(), accuracy.std(), ' is mean and std of accuracy rmse test')


 

		return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, rmse_train, rmse_test, accept_ratio)





def histogram_trace(pos_points, fname): # this is function to plot (not part of class)
		 
		size = 15

		plt.tick_params(labelsize=size)
		params = {'legend.fontsize': size, 'legend.handlelength': 2}
		plt.rcParams.update(params)
		plt.grid(alpha=0.75)

		plt.hist(pos_points,  bins = 20, color='#0504aa', alpha=0.7)   
		plt.title("Posterior distribution ", fontsize = size)
		plt.xlabel(' Parameter value  ', fontsize = size)
		plt.ylabel(' Frequency ', fontsize = size) 
		plt.tight_layout()  
		plt.savefig(fname + '_posterior.png')
		plt.clf()


		plt.tick_params(labelsize=size)
		params = {'legend.fontsize': size, 'legend.handlelength': 2}
		plt.rcParams.update(params)
		plt.grid(alpha=0.75) 
		plt.plot(pos_points)   

		plt.title("Parameter trace plot", fontsize = size)
		plt.xlabel(' Number of Samples  ', fontsize = size)
		plt.ylabel(' Parameter value ', fontsize = size)
		plt.tight_layout()  
		plt.savefig(fname  + '_trace.png')
		plt.clf()


 

def main():
  

	outres = open('results.txt', 'w')

 


	

	for problem in range(2, 3): 

 
  
 
		if problem == 1: #Single step ahead prediction
			traindata = np.loadtxt("data/Sunspot/train.txt")
			testdata = np.loadtxt("data/Sunspot/test.txt")  # 
			features = 4  #
			output = 1
			activation = True # true for sigmoid, false for softmax


		if problem == 2: #Multi-step ahead prediction. #MMM stock market - 5 steps ahead predicton for closing stock price https://au.finance.yahoo.com/quote/mmm/
			traindata = np.loadtxt("data/Stockmarket/train.txt")
			testdata = np.loadtxt("data/Stockmarket/test.txt")  # 
			features = 5  #
			output = 5
			activation = True  # true for sigmoid, false for softmax
 

 


		print(traindata)

		topology = [features, output]


		model = logistic_regression(500,   traindata, testdata, topology[0], 0.1, activation) 

		train_perc, test_perc, rmse_train, rmse_test = model.SGD()

		print(train_perc, test_perc, rmse_train, rmse_test, ' train_perc, test_perc, rmse_train, rmse_test using SGD ')


 
		#--------------------------------------------------

		MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)


		numSamples = 2000# need to decide yourself

		mcmc = MCMC(numSamples, traindata, testdata, topology, activation)  # declare class

		[pos_w, pos_tau, fx_train, fx_test,   rmse_train, rmse_test, accept_ratio] = mcmc.sampler() 


		fx_mu = fx_test.mean(axis=0)
		fx_high = np.percentile(fx_test, 95, axis=0)
		fx_low = np.percentile(fx_test, 5, axis=0)

		fx_mu_tr = fx_train.mean(axis=0)
		fx_high_tr = np.percentile(fx_train, 95, axis=0)
		fx_low_tr = np.percentile(fx_train, 5, axis=0)


		rmse_tr = np.mean(rmse_train)
		rmsetr_std = np.std(rmse_train)
		rmse_tes = np.mean(rmse_test)
		rmsetest_std = np.std(rmse_test)

		np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

		ytestdata = testdata[:, features]
		ytraindata = traindata[:, features]
		x_test = np.linspace(0, 1, num=testdata.shape[0])
		x_train = np.linspace(0, 1, num=traindata.shape[0])

		 

		'''plt.plot(x_test, ytestdata, label='actual')
		plt.plot(x_test, fx_mu, label='pred. (mean)')
		plt.plot(x_test, fx_low, label='pred.(5th percen.)')
		plt.plot(x_test, fx_high, label='pred.(95th percen.)')
		plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Test Data Uncertainty ")
		plt.savefig('mcmctest.png') 
		plt.clf()
		# -----------------------------------------
		plt.plot(x_train, ytraindata, label='actual')
		plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
		plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
		plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
		plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Train Data Uncertainty ")
		plt.savefig('mcmctrain.png') 
		plt.clf()'''

		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)

		ax.boxplot(pos_w)

		ax.set_xlabel('[w0] [w1] [w3] [b]')
		ax.set_ylabel('Posterior')

		plt.legend(loc='upper right')

		plt.title("Posterior")
		plt.savefig('w_pos.png') 

		plt.clf()

		import os
		folder = 'posterior'
		if not os.path.exists(folder):
			os.makedirs(folder)

		#for i in range(pos_w.shape[1]):

		for i in range(5):
			histogram_trace(pos_w[:,i], folder+ '/'+ str(i))

if __name__ == "__main__": main()


 