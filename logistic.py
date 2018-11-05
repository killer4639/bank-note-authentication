import numpy as np
import pandas as pan

np.seterr(over='ignore')

num_folds = 3
num_samples = 1371
num_features = 5
num_classes = 2
num_runs = 5
lr = 2
list_ratio = [.01,.02, .05 ,.1 ,.625, 1] 

data_file_name = "data.csv"

def logistic_reg():
	# reading the training data and storing it in array
	inp = pan.read_csv(data_file_name)
	data = np.array(inp)

	# initialising variables
	temp = np.ones((1,num_samples))
	x = np.array((num_samples,(num_features+1)))
	x = np.concatenate((temp.T,data),axis=1)

	weights = avg_weights = np.ones((np.size(list_ratio),num_features))
	accuracy = np.zeros((1,np.size(list_ratio)))
	
	# calculating indices and seperating training and validation set
	validate_index2 = (int)((num_samples/num_folds)-1)
	validate_index1 = (int)(validate_index2 - (num_samples/num_folds) + 1)
	training_data = np.vstack((x[:(np.absolute(validate_index1-1)),:],x[(validate_index2+1):,:]))
	validate_data = x[validate_index1:validate_index2, :]

	# looping over the ratios
	for ratio in range(np.size(list_ratio)):
		
		training_set = np.zeros(((int)(list_ratio[ratio]*training_data.shape[0]),(num_features+1)))
		
		# looping for number of runs over each ratio
		for runs in range(0,num_runs):
			weights[ratio,:] = 0

			# random selection of rows from training set
			training_set[:,:] = training_data[np.random.choice(training_data.shape[0],size = (int)(list_ratio[ratio]*training_data.shape[0]) , replace=False), :]
			
			# calling the training function which returns the weights
			weights[ratio,:] = training(training_set,weights[ratio,:])				
			
			# calling the validate function which returns the accuracy
			accuracy[0,ratio] += validate(validate_data,weights[ratio,:])
			avg_weights[ratio,:] += weights[ratio,:]	
		
		# average of accuracies and weights obtained for each training sample ratio	
		avg_weights[ratio,:] /= num_runs
		accuracy[0,ratio] /= num_runs
		print(accuracy[0,ratio])
		
	return (accuracy,avg_weights)			


# logistic training function
def training(training_set,weights):
	
	# seperating class and features
	y_training = np.zeros((training_set.shape[0],1))
	y_training[:,0] = training_set[:,num_features]

	temp = training_set.shape[0]
	training_set = np.delete(training_set,num_features,1)
	np.reshape(weights,(1,num_features))
	increment = np.zeros((num_features))

	# updating the weights for 1500 cycles. can be increased for better accuracy
	for i in range(1500):
		increment[:] = 0		
		for i in range(training_set.shape[0]):
			increment +=  (y_training[i,0]-predictor(training_set[i,:],weights))*training_set[i,:]
		weights += lr*increment
	
	return weights

# predictor function which calculates the sigmoid function value
def predictor(inp,weights):
	result = 1/(1+np.exp(-(weights.T.dot(inp))))	
	return result

# validate function takes validate data set, weights and returns accuracy
def validate(validate_data,weights):
	# seperating class and features
	y_validate = np.zeros((validate_data.shape[0],1))
	y_validate[:,0] = validate_data[:,num_features]

	temp = validate_data.shape[0]
	validate_data = np.delete(validate_data,num_features,1)
	np.reshape(weights,(1,num_features))
	num_crct_pred = 0
	
	for length in range(0,(int)(np.size(validate_data)/num_features)):
		result = predictor(validate_data[length,:],weights)
	
	# checking values of predictor function and comparing it with actual class value to find no:of correct predictions
		if result <= 0.5:
			if y_validate[length,0] == 0:
				num_crct_pred += 1
		else :
			if y_validate[length,0] == 1:
				num_crct_pred += 1

	accuracy = num_crct_pred/(np.size(validate_data)/num_features)
	
	return accuracy	
