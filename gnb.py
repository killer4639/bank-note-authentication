import numpy as np
import pandas as pan

num_folds = 3
num_samples = 1371
num_features = 5
num_classes = 2
num_runs = 5
lr = 2
list_ratio = [.01,.02, .05 ,.1 ,.625, 1] 

data_file_name = "data.csv"

# function to train data using guassian naive bayes
def guassian_naive_bayes():

	# reading input, storing it in array nad initialising variables
	inp = pan.read_csv(data_file_name)
	data = np.array(inp)
	means = means_avg = np.zeros((np.size(list_ratio),num_classes,num_features-1))
	variance = variance_avg = np.zeros((np.size(list_ratio), num_classes,num_features-1))
	means[:,:] = 0
	variance[:,:] = 0
	accuracy = np.zeros((1,np.size(list_ratio)))

	# calculating indices for validate data and training data		
	validate_index2 = (int)((num_samples/num_folds)-1)
	validate_index1 = (int)(validate_index2 - (num_samples/num_folds) + 1)

	# extracting validate data and training data
	training_data = np.vstack((data[:(np.absolute(validate_index1-1)),:],data[(validate_index2+1):,:]))
	validate_data = data[validate_index1:validate_index2, :]

	# looping over the list of training sampling ratios
	for ratio in range(np.size(list_ratio)):

		training_set = np.zeros(((int)(list_ratio[ratio]*training_data.shape[0]),(num_features)))

		# looping for no:of runs
		for runs in range(0,num_runs):
			means[ratio,:] = 0
			variance[ratio,:] = 0

			# extracting random rows from training set for each ratio
			training_set[:,:] = training_data[np.random.choice(training_data.shape[0],size = (int)(list_ratio[ratio]*training_data.shape[0]) , replace=False), :]
			
			# calculating guassian means for the independent parameters of features
			for j in range(0,num_classes):
				for k in range(0,num_features-1):
					means[ratio,j,k] = gnb_mean_esti(j,k,training_set)
			means_avg[ratio,j,k] += means[ratio,j,k]

			# calculating guassian variances for independent parameters of features
			for j in range(0,2):
				for k in range(0,4):
					variance[ratio,j,k] = gnb_var_esti(j,k,training_set,means[ratio,:,:])
				variance_avg[ratio,j,k] += variance[ratio,j,k]

			# validating the parameters and finding accuracy for each sampling ratio	
			accuracy[0,ratio] += gnb_validate(validate_data,means[ratio,:,:],variance[ratio,:,:])
		
		# averaging the accuracies
		accuracy[0,ratio] /= num_runs
		print(accuracy[0,ratio])

	return (accuracy,means_avg,variance_avg)

# validate function which takes validation data, guassian means and variances and return accuracy
def gnb_validate(validate_data,means,variance):
	
	num_pos_samples = 0

	# finding no:of positive nad negative samples
	for i in range(0,validate_data.shape[0]):		
		num_pos_samples += validate_data[i,num_features-1]
	num_neg_samples = validate_data.shape[0]-num_pos_samples

	# finding bernoulli parameter of class variable
	py0 = num_neg_samples/(num_neg_samples+num_pos_samples)
	py1 = 1-py0
	num_crct_pred = 0	

	# looping over the entire validation data
	for length in range(0,validate_data.shape[0]):
		# variables for likelihood probability estimation
		prod_pxy0 = prod_pxy1 = 1

		# finding the likelihood probabilities
		for i in range(0,num_features-1):
			prod_pxy0 *= calculate_guassian(validate_data[length,i],means[0,i],variance[0,i])
			prod_pxy1 *= calculate_guassian(validate_data[length,i],means[1,i],variance[1,i])
		
		# finding the posterior probabilities, predicting class and comparing with actual value of class to find accuracy
		py0x = (prod_pxy0*py0)/(prod_pxy0*py0 + prod_pxy1*py1)
		if py0x <= 0.5:
			if validate_data[length,4] == 1:
				num_crct_pred += 1
		else :
			if validate_data[length,4] == 0:
				num_crct_pred += 1

	accuracy = num_crct_pred/(validate_data.shape[0])

	return accuracy

# function to return the guassian probability
def calculate_guassian(x,mean,variance):
	result = (np.exp(-(np.power(x-mean,2)/variance)))/(np.sqrt(2*3.14*variance))
	return result

# function for estimating the means of features
def gnb_mean_esti(class_type, feature,training_set):
	mean = 0
	num_pos_samples = 0
	
	# extracing no:of positive and negative samples to find means
	for i in range(0,training_set.shape[0]):		
		num_pos_samples += training_set[i,num_features-1]		

		if training_set[i,num_features-1] == class_type:
			mean += training_set[i,feature]

	# checking class type and calculating means
	if (class_type):
		mean = mean/num_pos_samples
	else:
		mean = mean/(training_set.shape[0]-num_pos_samples)
	return mean

# function for estimating variances of features 
def gnb_var_esti(class_type,feature,training_set,means):
	variance = 0
	num_pos_samples = 0
	
	# extracing no:of positive and negative samples to find variances
	for i in range(0,training_set.shape[0]):
		num_pos_samples += training_set[i,num_features-1]

		if training_set[i,num_features-1] == class_type:
			variance += np.power((training_set[i,feature]-means[class_type,feature]),2)	
	
	# checking class type and calculating unbiased variances
	if (class_type):
		variance = variance/(num_pos_samples-1)
	else:
		variance = variance/((training_set.shape[0]-num_pos_samples)-1)	

	return variance
