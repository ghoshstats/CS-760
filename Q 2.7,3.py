import matplotlib.pyplot as plt
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, label=None, left=None, right=None):
        self.feature_index = feature_index  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.label = label  # Class label for a leaf node
        self.left = left  # Left subtree for instances where feature < threshold
        self.right = right  # Right subtree for instances where feature >= threshold


# Redefine the function to calculate entropy
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# Redefine the function to build the decision tree
def build_tree(data, depth=0):
    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    
    # If the node is pure (entropy is zero), return a leaf node with the label
    if calculate_entropy(y) == 0:
        return DecisionTreeNode(label=y[0])
    
    # If there are no candidate splits (no feature/threshold improves information gain), return a leaf node with the majority label
    if len(np.unique(X, axis=0)) == 1:
        unique_label, count = np.unique(y, return_counts=True)
        majority_label = unique_label[np.argmax(count)]
        return DecisionTreeNode(label=majority_label)
    
    best_gain = 0
    best_feature_index = None
    best_threshold = None
    best_left = None
    best_right = None
    
    # Loop over each feature and each possible threshold to find the best split
    for feature_index in range(X.shape[1]):
        for threshold in np.unique(X[:, feature_index]):
            left_indices = X[:, feature_index] < threshold
            right_indices = X[:, feature_index] >= threshold
            
            left_y = y[left_indices]
            right_y = y[right_indices]
            
            # If the split results in empty nodes, skip this split
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            
            # Calculate the information gain of the split
            p_left = len(left_y) / len(y)
            p_right = len(right_y) / len(y)
            gain = calculate_entropy(y) - p_left * calculate_entropy(left_y) - p_right * calculate_entropy(right_y)
            
            # If this split is the best so far, update the best split parameters
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold
                best_left = (X[left_indices], left_y)
                best_right = (X[right_indices], right_y)
    
    # If no valid split is found, return a leaf node with the majority label
    if best_gain == 0:
        unique_label, count = np.unique(y, return_counts=True)
        majority_label = unique_label[np.argmax(count)]
        return DecisionTreeNode(label=majority_label)
    
    # Recursively build the left and right subtrees
    left_tree = build_tree(list(zip(*best_left)))
    right_tree = build_tree(list(zip(*best_right)))
    
    return DecisionTreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left_tree, right=right_tree)


# Redefine the function to count the number of nodes in the tree
def count_nodes(tree):
    if tree is None:
        return 0
    if tree.label is not None:
        return 1  # Leaf node
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)  # Internal node + left subtree + right subtree


# Redefine the function to predict the label of a data point using the decision tree
def predict(tree, point):
    if tree.label is not None:  # If the node is a leaf node
        return tree.label
    # Traverse the tree based on the feature value of the point
    if point[tree.feature_index] < tree.threshold:
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)


# Redefine the function to calculate the error rate of the decision tree on a dataset
def calculate_error_rate(tree, data):
    incorrect_predictions = sum([predict(tree, point) != label for point, label in data])
    return incorrect_predictions / len(data)


# Now, let's rerun the process to train decision trees and plot the learning curve using the custom implementation.
num_nodes_list_custom = []
error_rate_list_custom = []

# Iterate over each training size, train a decision tree, and calculate its error rate on the test set
for size in training_sizes:
    # Generate the nested training set Dn
    training_set = training_set_dbig[:size]
    
    # Train a decision tree on Dn using the custom implementation
    tree_custom = build_tree(training_set)
    
    # Calculate the number of nodes in the tree
    num_nodes_custom = count_nodes(tree_custom)
    
    # Calculate the error rate of the tree on the test set using the custom implementation
    error_rate_custom = calculate_error_rate(tree_custom, test_set_dbig)
    
    # Append the results to the lists
    num_nodes_list_custom.append(num_nodes_custom)
    error_rate_list_custom.append(error_rate_custom)

# Organize the results in a structured format (table)
results_table_custom = {
    "n": training_sizes,
    "Number of Nodes": num_nodes_list_custom,
    "Error Rate": error_rate_list_custom
}
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, error_rate_list_custom, marker='o')
plt.xscale('log')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Test Set Error Rate')
plt.title('Learning Curve (Manually Implemented)')
plt.grid(True)
plt.show()

results_table_custom

# Define the function to plot the decision boundary using the custom implementation
def plot_decision_boundary_custom(training_set, tree, title):
    # Extract the data points and labels from the training set
    X, y = zip(*training_set)
    X = np.array(X)
    y = np.array(y)

    # Define the plot boundaries
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Generate a grid of points within the plot boundaries
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict the labels for each point in the grid
    Z = np.array([predict(tree, point) for point in grid_points])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the training points
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()


# Plot the decision boundaries for each training set size using the custom implementation
for size in training_sizes:
    plot_decision_boundary_custom(training_set_dbig[:size], build_tree(training_set_dbig[:size]), f'Training Size = {size} (Custom Implementation)')

################################################## Q3 (Using sklearn) ####################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Define a function to visualize the decision boundaries of the decision tree
def plot_decision_boundary_sklearn(clf, X, y, title):
    plt.figure(figsize=(8, 6))
    
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='bwr')
    
    # Define the axis limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    # Create a mesh grid and predict the class labels for each point in the grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    
    plt.title(title)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    
    plt.show()

# Extract the feature matrix X and the target vector y from the training and test sets
X_train_dbig = np.array([point for point, _ in training_set_dbig])
y_train_dbig = np.array([label for _, label in training_set_dbig])
X_test_dbig = np.array([point for point, _ in test_set_dbig])
y_test_dbig = np.array([label for _, label in test_set_dbig])

# Re-initialize the lists to store the results
num_nodes_list = []
error_rate_list = []

# Iterate over each training size, train a decision tree using scikit-learn, and calculate its error rate on the test set
for size in training_sizes:
    # Generate the nested training set Dn
    X_train = X_train_dbig[:size]
    y_train = y_train_dbig[:size]
    
    # Initialize and train the DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    
    # Calculate the number of nodes in the tree
    num_nodes = clf.tree_.node_count
    
    # Calculate the error rate of the tree on the test set
    y_pred = clf.predict(X_test_dbig)
    error_rate = 1 - accuracy_score(y_test_dbig, y_pred)
    
    # Append the results to the lists
    num_nodes_list.append(num_nodes)
    error_rate_list.append(error_rate)
    
    # Plot the decision boundary
    plot_decision_boundary_sklearn(clf, X_train, y_train, f'Training Size = {size}')

# Organize the results in a structured format (table)
results_table = {
    "n": training_sizes,
    "Number of Nodes": num_nodes_list,
    "Error Rate": error_rate_list
}

# Plot the learning curve (n vs. err_n)
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, error_rate_list, marker='o')
plt.xscale('log')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Test Set Error Rate')
plt.title('Learning Curve (using sklearn)')
plt.grid(True)
plt.show()

results_table
