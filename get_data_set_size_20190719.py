import datasets

tasks = [[0,1], [2,3], [4,5], [6,7], [8,9]]

for t in tasks:
	data = datasets.split_mnist(t, [])
	# data = datasets.split_fashion_mnist(t, [])
	#train_data, train_labels = data.get_train_samples()
	#print("Size of data set {} is {}".format(t, train_data.shape[0]))
	[test_data, test_labels] = data.get_eval_samples()
	print("Size of data set {} is {}".format(t, test_data.shape[0]))

tasks = [[0],[1], [2],[3], [4],[5], [6],[7], [8],[9]]

for t in tasks:
	data = datasets.split_mnist(t, [])
	# data = datasets.split_fashion_mnist(t, [])
	#train_data, train_labels = data.get_train_samples()
	#print("Size of data set {} is {}".format(t, train_data.shape[0]))
	[test_data, test_labels] = data.get_eval_samples()
	print("Size of data set {} is {}".format(t, test_data.shape[0]))

