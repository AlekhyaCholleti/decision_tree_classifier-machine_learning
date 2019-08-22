import pandas as pd

df = pd.read_csv('/home/welcome/Downloads/ml_ai_dl/iris.csv')
#shuffling all the rows in the df
df = df.sample(frac=1).reset_index(drop=True)
#converting the df into list of lists
data_list = df.values.tolist()

training_data = data_list[:120]
testing_data = data_list[120:150]

header = ["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"]


def unique_values(data, col):
	"""Find the unique values for a column in a dataset."""
	return set((row[col] for row in data))
#print(unique_values(training_data, 0))	 


def class_counts(data):
	"""Counts the number of each type of class in a dataset."""
	counts = {}
	for row in data:
		label = row[-1]
		if label not in counts:
			counts[label] = 0
		counts[label] += 1
	return counts	
#print(class_counts(training_data))	


def is_numeric(value):
	"""Test if a value is numeric."""
	return isinstance(value, int) or isinstance(value, float)
#print(is_numeric("red"))    


class question():
	"""A Question is used to partition a dataset.

	This class just records a 'column number' (e.g., 1 for Sepal.width) and a
	'column value' (e.g., 1). The 'match' method is used to compare
	the feature value in an example to the feature value stored in the
	question"""	

	def __init__(self, col, value):
		self.col = col
		self.value = value

	def match(self,row):
		row_value = row[self.col]
		if(is_numeric(row_value)):
			return row_value >= self.value
		else:
			return row_value == self.value

	def __repr__(self):
		if(is_numeric(self.value)):
			return ("is %s >= %s ") %(header[self.col],self.value)
		else:
			return  ("is %s == %s ") %(header[self.col],self.value)
	

#print(question(1,1)) 
#print(question(1,1).match(training_data[0]))


def partition(data,q):
	"""Partitions a dataset.

	For each row in the dataset, check if it matches the question. If
	so, add it to 'true rows', otherwise, add it to 'false rows'.
	"""	

	true_data = [] 
	false_data = []
	for row in data:
		if(q.match(row)):
			true_data.append(row)
		else:
			false_data.append(row)	
	return true_data, false_data
#que = question(0, "Green")
#true_data , false_data = partition(training_data,que)
#print(len(true_data))
#print(false_data)	


def gini(data):
	"""Calculate the Gini Impurity/uncertainity for a list of rows."""	

	G_I = 0
	classes_present = class_counts(data)
	total_count = 0
	for label_value in classes_present.values():
		total_count += label_value
	for label_value in classes_present.values():
		G_I += (label_value * (total_count - label_value))/(total_count ** 2)

	return G_I	
#print(gini(true_data))
#print(gini(false_data))


def info_gain(left_data, right_data,initial_gini):
	"""Information Gain.

	The uncertainty of the parent node, minus the weighted impurity of
	two child nodes.
	"""

	p = len(left_data) / (len(left_data) + len(right_data))
	return initial_gini - (p * gini(left_data)) - ((1 - p)* (gini(right_data)))
#g = gini(training_data)
#print(info_gain(true_data, false_data, g))	


def best_split(data):
	"""Find the best question to ask by iterating over every column and every value in it
	by calculating the information gain.(should be high)"""

	best_gain = 0
	best_question = None
	noof_features = len(data[0]) - 1

	for col in range(0,noof_features):
		values = unique_values(data, col)

		for value in values:
			que = question(col, value)
			true_data , false_data = partition(data, que)
			if((len(true_data)) == 0 or (len(false_data) == 0)):
				continue
			gain = info_gain(true_data,false_data,gini(data))
			if(gain >= best_gain):
				best_gain = gain
				best_question = que
	return best_gain,best_question			
#best_gain,best_question = best_split(training_data)
#print(best_gain)
#print(best_question)


class leaf():
	"""A Leaf node classifies data.

	This holds a dictionary of class (e.g., "Setosa") -> number of times
	it appears in the rows from the training data that reach this leaf.
	"""

	def __init__(self,data):
		self.predictions = class_counts(data)


class decision_node():
	"""A Decision Node asks a question.

	This holds a reference to the question, and to the two child nodes.
	"""

	def __init__(self,question,true_branch,false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch

def build_tree(data):
	"""Builds the tree."""

	gain , question = best_split(data)
	if gain == 0:
		return leaf(data)
	true_data ,false_data = partition(data,question)
	true_branch = build_tree(true_data)
	false_branch = build_tree(false_data)
	return decision_node(question,true_branch,false_branch)

def print_tree(node, spacing=""):

	if isinstance(node, leaf):
		print (spacing + "Predict", node.predictions)
		return

	print (spacing + str(node.question))

	print (spacing + '--> True:')
	print_tree(node.true_branch, spacing + "  ")

	print (spacing + '--> False:')
	print_tree(node.false_branch, spacing + "  ")	
my_tree = build_tree(training_data)
print_tree(my_tree)

def classify(row, node): 
	"""a helping function
	  #tells under what leaf the row/sample comes"""
	if isinstance(node,leaf):
		return node.predictions
	if node.question.match(row) == True:
		return  classify(row, node.true_branch)	
	else:
		return  classify(row, node.false_branch)	
#print(classify(training_data[4],my_tree))		

def print_leaf(counts):
	"""a helping function
	  #ta nicer way to print the predictions at a leaf"""
	total = sum(counts.values()) * 1.0
	probs = {}
	for lbl in counts.keys():
		probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
	return probs
#print(print_leaf(classify(training_data[70],my_tree)))    



for row in testing_data:
	print ("Actual: %s. Predicted: %s" %
		   (row[-1], print_leaf(classify(row, my_tree))))
