import importlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pan
import numpy as np

# importing the logistic regression and guassian naive bayes funcitons as a module
import logistic
import gnb

num_folds = 3
num_samples = 1371
num_features = 5
num_classes = 2
list_ratio = [.01,.02, .05 ,.1 ,.625, 1]

#initialising variables to store the trained parameters
gnb_accuracy = np.array(np.size(list_ratio))  # to store the accuracy of gaussian naive bayes model
logistic_accuracy = accuracy = np.zeros((1,np.size(list_ratio)))	# to store the accuracy of logistic regression model
gnb_means = np.zeros((np.size(list_ratio),num_classes,num_features-1))	# to store the guassian means of gaussian naive bayes model
gnb_variance = np.zeros((np.size(list_ratio),num_classes,num_features-1))	# to store the guassian variances of gaussian naive bayes model
logistic_weights = np.ones((np.size(list_ratio),num_features))	# to store the weights of logistic regression model

# calling the training funcitons of individual models
(logistic_accuracy,logistic_weights) = logistic.logistic_reg()
(gnb_accuracy,gnb_means,gnb_variance) = gnb.guassian_naive_bayes()

# converting accuracies to percentage scale
gnb_accuracy *= 100
logistic_accuracy *= 100

gnb_accuracy = np.reshape(gnb_accuracy,6)
logistic_accuracy = np.reshape(logistic_accuracy,6)

print("\n means of gnb model for different ratios:\n",gnb_means, "\n variances of gnb model:\n", gnb_variance)
print("\n accuracies of gnb model for different ratios:\n",gnb_accuracy)
print("\n weights of logistic regression model for different ratios:\n",logistic_weights)
print("\n accuracies of logistic model for different ratios:\n",logistic_accuracy)


# plotting the graphs on same plane for sampling_ratio vs accuracy
plt.plot(list_ratio,logistic_accuracy,color = 'g')
plt.plot(list_ratio,gnb_accuracy,color = 'orange' )
plt.xlabel('training_sample_ratio')
plt.ylabel('validation accuracy in percentage')
green_patch = mpatches.Patch(color='green', label='logistic regression')
orange_patch = mpatches.Patch(color='orange', label='guassian naive bayes')
plt.legend(handles=[green_patch,orange_patch])
plt.show()

rows = 613
columns = num_features-1
guassian_samples = np.zeros((rows,columns))
means_guassian_samples = np.zeros((1,num_features-1))

# taking the parameters obtained by training the data set with guassian naive bayes model and extracting random samples using numpy.random.normal
for i in range(0,num_features-1):
	temp = np.random.normal(gnb_means[5,1,i],np.sqrt(gnb_variance[5,1,i]),rows)
	guassian_samples[:,i] = temp
	# calculating means of randomly extracted samples using trained parameters
	means_guassian_samples[0,i] = np.mean(guassian_samples[:,i])

np.savetxt('samples.csv',guassian_samples,delimiter=',',newline='\n',header='GUASSIAN SAMPLES')

print("\n means obtained by training data set:\n ",gnb_means[5,1,:], "\n means of guassian samples:\n", means_guassian_samples)
