import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib as mb
from pathlib import Path
import time

K = 10
d = 3072

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_one_hot(index, array_size = 10):
    one_hot = [0] * array_size
    one_hot[index] = 1
    return one_hot


def display_imgs(file):
    global K, d
    batch_data = unpickle(file)
    X, Y, y = [], [], []
    for i, img in enumerate(batch_data[b"data"]):
        # if i > 0:
        #     break
        reshaped_img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        plt.imshow(reshaped_img)


def read_imgs(file):
    global K, d
    batch_data = unpickle(file)
    X = np.array(normalize_distribution(batch_data[b"data"].T))
    y = np.array(batch_data[b"labels"])
    Y = np.array(np.eye(K)[y].T)
    return X, Y, y

def normalize_distribution(X):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X = (X - mean_X) / std_X
    # X = X - mean_X[:, np.newaxis]
    # X = X / std_X[:, np.newaxis]
    # print(X)
    return X

def initialize_parameters():
    global K, d
    np.random.seed(400)
    W = np.random.normal(loc = 0.0, scale = 0.01, size = (K, d))
    b = np.random.normal(loc = 0.0, scale = 0.01, size = (K, 1))
    return W, b

def evaluate_classifier(X, W, b):
    """
    X: input data
    W: weights for each input data point
    b: bias parameter
    Get the outputs (Y) for the classifier.
    We use the linear function Y = W.X + b
    """
    s = np.dot(W, X) + b
    P = softmax(s)
    return P

def softmax(s):
    """
    Activation function that turns our probabilites into values summing to 1.
    Returns all new values for the probabilities.
    """
    e_x = np.exp(s - np.max(s))
    return e_x / e_x.sum(axis=0) 

def compute_loss(X, Y, W, b, lambda_value):
    """
    Computes the cross-entropy loss of the NN.
    It uses the cross-entropy formulae for each input as well as a
    regularization term.
    """
    P = evaluate_classifier(X, W, b)

    cross_entropy_loss = (-1 / X.shape[1]) * np.sum(
        [np.log(np.dot(Y[:, index].T, P[:, index])) for index in range(X.shape[1])]
    )
    regularization_term = lambda_value * np.sum(np.square(W))
    total_loss = cross_entropy_loss + regularization_term
    return total_loss
    

def compute_accuracy(X, y, W, b):
    """
    Computes the accuracy of the classifier for a given set of examples:
    percentage of examples for which it gets the correct answer.
    - Each column of X corresponds to an image and X has size dxn.
    - y is the vector of ground truth labels of length n.
    - acc is a scalar value containing the accuracy.
    k ∗ = arg max {p1, ..., pK}
    """
    P = evaluate_classifier(X, W, b)

    # Returns the column-index of the max value within each row of P (predicted scores)
    pred_class_labels = np.argmax(P, axis=0)
    # print('Predicted outputs', pred_class_labels)
    # Computes and returns the percentage of correctly predicted labels
    return np.mean(pred_class_labels == y) * 100


def compute_gradient(X, Y, P, W, b, lambda_value):
    """
    -each column of X corresponds to an image and it has size d×n.
    -each column of Y (K×n) is the one-hot ground truth label for the cor-
    responding column of X.
    -each column of P contains the probability for each label for the image
    in the corresponding column of X. P has size K×n.
    -grad W is the gradient matrix of the cost J relative to W and has size
    K×d. Also called dw.
    -grad b is the gradient vector of the cost J relative to b and has size
    K×1.
    """
    n_batch = X.shape[1]
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)
    divider = 1 / n_batch
    # Gbatch (dz)
    G_batch = -1 * (Y - P)
    # Computing W gradient
    grad_W = divider * np.dot(G_batch, X.T)
    # Add regularization parameter
    grad_W += 2 * lambda_value * W
    # Computing bias gradient
    grad_b = divider * np.dot(G_batch, np.ones((n_batch, 1)))

    return grad_W, grad_b

def ComputeGradsNumSlow(X, Y, P, W, b, lambda_value, h):
	""" Converted from matlab code by Andree Hultgren """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = compute_loss(X, Y, W, b_try, lambda_value)

		b_try = np.array(b)
		b_try[i] += h
		c2 = compute_loss(X, Y, W, b_try, lambda_value)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = compute_loss(X, Y, W_try, b, lambda_value)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = compute_loss(X, Y, W_try, b, lambda_value)

			grad_W[i,j] = (c2-c1) / (2*h)

	return grad_W, grad_b

def ComputeGradsNum(X, Y, P, W, b, lambda_value, h):
	""" Converted from matlab code by Andree Hultgren """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = compute_loss(X, Y, W, b, lambda_value)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = compute_loss(X, Y, W, b_try, lambda_value)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = compute_loss(X, Y, W_try, b, lambda_value)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def grad_checking(dw, dw2, db, db2):
    check_dW = ( np.abs(dw[0, 0] - dw2[0, 0]) ) / ( max(1e-6, np.abs(dw[0, 0]) + np.abs(dw2[0, 0])) )
    check_db = ( np.abs(db[0, 0] - db2[0, 0]) ) / ( max(1e-6, np.abs(db[0, 0]) + np.abs(db2[0, 0])) )
    print('Check dW: ', check_dW)
    print('Check db: ', check_db)
    return check_dW < 1e-6 and check_db < 1e-6


def create_mini_batches(X_train, Y_train, n_batch):

    indexes = np.random.permutation(X_train.shape[1])
    X_train = X_train[:, indexes]
    Y_train = Y_train[:, indexes]
    mini_batches = []
    for i in range(0, X_train.shape[1], n_batch):
        X_batch = X_train[:, i:i + n_batch]
        Y_batch = Y_train[:, i:i + n_batch]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches

def minibatch_gradient_descent(X, Y, y_labels, GD_params, W, b, lambda_value):
    # GD_params: dictionary
    n_batch = GD_params['n_batch']   # Size of the mini-batches.
    eta = GD_params['eta']     # Learning rate.
    n_epochs = GD_params['n_epochs']   # One epoch is a complete run through all the training samples.
    loss_history = []
    accuracy_history = []

    for epoch in range(n_epochs):
        mini_batches = create_mini_batches(X, Y, n_batch)
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            P = evaluate_classifier(X_mini, W, b)
            dW, db = compute_gradient(X_mini, Y_mini, P, W, b, lambda_value)
            W -= eta * dW
            b -= eta * db
        loss = compute_loss(X, Y, W, b, lambda_value)
        loss_history.append(loss)
        # print('Loss: ', loss)
        accuracy = compute_accuracy(X, y_labels, W, b)
        accuracy_history.append(accuracy)
        # print('Accuracy: ', accuracy)
    
    return loss_history, accuracy_history, W, b

def plot_loss(n_epochs, loss_history_train, loss_history_val, 
    accuracy_history_train, accuracy_history_val):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    range_epochs = range(1, n_epochs + 1)
    
    if loss_history_train:
        ax1.plot(range_epochs, loss_history_train, label='Training set')
    if loss_history_val:
        ax1.plot(range_epochs, loss_history_val, label='Validation set')
    if accuracy_history_train:
        ax2.plot(range_epochs, accuracy_history_train, label='Training set')
    if accuracy_history_val:
        ax2.plot(range_epochs, accuracy_history_val, label='Validation set')

    fig.suptitle('NN Performance', x=0.25)
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss')
    ax1.grid()
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    ax2.grid()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    fig.savefig('imgs/graph_case_4.png')
    plt.show()


def plot_learnt_weight_matrix(W):
    global K
    class_templates = []
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(W.shape[0]):
        plt.subplot(2, 5, i + 1)
        img = np.transpose(np.reshape(W[i, :], (3, 32, 32)), (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
        class_templates.append(img)
        plt.axis('off')
        plt.imshow(img)
        plt.title(class_labels[i])
    # Plot big-sized labeled images
    plt.suptitle('Weigth Matrix Templates')
    plt.savefig('imgs/labeled_templates_case_4.png')
    plt.show()
    # Plot all images concatenated
    plt.title('Weigth Matrix Templates')
    plt.imshow(np.concatenate(tuple(class_templates), axis=1))
    plt.axis('off')
    plt.savefig('imgs/templates_case_4.png')
    plt.show()

if __name__ == "__main__":
    data_path = '../datasets/cifar-10-batches-py/'
    batch_file_training = data_path + 'data_batch_1'
    batch_file_validation = data_path + 'data_batch_2'
    batch_file_test = data_path + 'test_batch'
    init_time = time.time()
    num_data = 10000

    # Initializing parameters
    W, b = initialize_parameters()
    GD_params = {'lambda': 1, 'n_epochs': 40, 'n_batch': 100, 'eta': 0.001}

    # Reading training set
    X_train, Y_train, y_train = read_imgs(batch_file_training)
    # X_train = X_train[:, :num_data]
    # Y_train = Y_train[:, :num_data]
    # y_train = y_train[:num_data]

    # Reading validation set
    X_val, Y_val, y_val = read_imgs(batch_file_validation)
    # X_val = X_val[:, :num_data]
    # Y_val = Y_val[:, :num_data]
    # y_val = y_val[:num_data]

    # Reading test set
    X_test, Y_test, y_test = read_imgs(batch_file_test)

    ########## Block for Gradient Checking ##########
    #
    # P = evaluate_classifier(X_train, W, b)
    # dW, db = compute_gradient(X_train, Y_train, P, W, b, GD_params['lambda'])
    # dW_center_diff, db_center_diff = ComputeGradsNumSlow(X_train, \
    #     Y_train, P, W, b, GD_params['lambda'], 1e-6)
    # dW_finite, db_finite = ComputeGradsNum(X_train, \
    #     Y_train, P, W, b, GD_params['lambda'], 1e-6)

    # print('Analytical vs Centered Difference:', grad_checking(dW, dW_center_diff, db, db_center_diff))
    # print('Analytical vs Finite Difference:', grad_checking(dW, dW_finite, db, db_finite))
    #
    ########## End of Gradient Checking ##########

    ########## Block for Training and Validation ##########
    #
    # Training
    loss_history_train, accuracy_history_train, W_star, b_star = \
        minibatch_gradient_descent(X_train, Y_train, y_train, GD_params, W, b, GD_params['lambda'])
    # Validation
    loss_history_val, accuracy_history_val, W_star_val, b_star_val = \
        minibatch_gradient_descent(X_val, Y_val, y_val, GD_params, W_star, b_star, GD_params['lambda'])
    #
    ########## End of Training and Validation ##########

    final_time = time.time()
    
    ########## Block for Final Testing ##########
    #
    testing_accuracy = compute_accuracy(X_test, y_test, W_star, b_star)
    #
    ########## End of Testing ##########
    
    print('Time for training and validating: ', final_time - init_time, 'sec')
    print('Training -> ', 'Initial Loss: ', loss_history_train[0], 'Final Loss: ',
        loss_history_train[-1], 'Accuracy: ', accuracy_history_train[-1])
    
    print('Validation -> ', 'Initial Loss: ', loss_history_val[0], 'Final Loss: ',
        loss_history_train[-1], 'Accuracy: ', accuracy_history_val[-1])

    plot_loss(GD_params['n_epochs'], loss_history_train, loss_history_val, accuracy_history_train, 
        accuracy_history_val)

    plot_learnt_weight_matrix(W_star)

    print('Final Accuracy in Testing: ', testing_accuracy)