from string import punctuation, digits
import numpy as np
import random
import utils

# Part I


def get_order(n_samples) :
    try :
        with open(str(n_samples) + '.txt') as fp :
            line = fp.readline()
            return list(map(int, line.split(',')))

    except FileNotFoundError :
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0) :
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.
    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    y = label * (np.dot(theta, feature_vector) + theta_0)

    if y < 1 :
        hinge = 1 - y
    if y >= 1 :
        hinge = 0

    return hinge
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0) :
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here

    sum = 0

    for row, y in zip(feature_matrix, labels) :
        loss = hinge_loss_single(row, y, theta, theta_0)
        sum += loss

    return sum / len(feature_matrix)
    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0) :

    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    value = label * (np.dot(current_theta, feature_vector) + current_theta_0)

    if value <= 0 :
        new_theta = current_theta + label * feature_vector
        new_theta_0 = current_theta_0 + label
        return (new_theta, new_theta_0)

    else :
        return (current_theta, current_theta_0)

    raise NotImplementedError


def perceptron(feature_matrix, labels, T) :
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    n = feature_matrix.shape[1]
    theta = np.zeros(n)
    theta_0 = 0

    for t in range(T) :

        for i in get_order(feature_matrix.shape[0]) :
            # Your code here
            conv = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta = conv[0]
            theta_0 = conv[1]
            pass

    return (theta, theta_0)
    raise NotImplementedError


def average_perceptron(feature_matrix, labels, T) :
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    n = feature_matrix.shape[1]
    m = feature_matrix.shape[0]
    theta = np.zeros(n)
    theta_0 = 0
    sums_of_theta = np.zeros(n)
    sums_of_theta_0 = 0

    for t in range(T) :

        for i in get_order(feature_matrix.shape[0]) :
            # Your code here
            conv = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta = conv[0]
            theta_0 = conv[1]
            sums_of_theta += theta
            sums_of_theta_0 += theta_0
            pass

    average_theta = np.divide(sums_of_theta, m * T)
    average_theta_0 = np.divide(sums_of_theta_0, m * T)

    return (average_theta, average_theta_0)
    raise NotImplementedError


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0) :
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    value = label * (np.dot(current_theta, feature_vector) + current_theta_0)

    if value <= 1 :
        new_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        new_theta_0 = current_theta_0 + eta * label
        return (new_theta, new_theta_0)

    else :
        new_theta = (1 - eta * L) * current_theta
        return (new_theta, current_theta_0)

    raise NotImplementedError


def pegasos(feature_matrix, labels, T, L) :
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lambda value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    n = feature_matrix.shape[1]
    theta = np.zeros(n)
    theta_0 = 0
    counter = 0

    for t in range(T) :

        for i in get_order(feature_matrix.shape[0]) :
            eta_t = 1 / (np.sqrt(counter + 1))
            final = pegasos_single_step_update(feature_matrix[i, :], labels[i], L, eta_t, theta, theta_0)
            counter += 1
            theta = final[0]
            theta_0 = final[1]

    return (theta, theta_0)
    raise NotImplementedError

# Part II


def classify(feature_matrix, theta, theta_0) :
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    prediction = np.dot(theta, np.transpose(feature_matrix)) + theta_0

    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1

    return prediction

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs) :
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    train_theta, train_theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_predictions = classify(train_feature_matrix, train_theta, train_theta_0)
    train_accuracy = accuracy(train_predictions, train_labels)

    #val_theta, val_theta_0 = classifier(val_feature_matrix, val_labels, **kwargs)
    val_predictions = classify(val_feature_matrix, train_theta, train_theta_0)
    val_accuracy = accuracy(val_predictions, val_labels)

    return(train_accuracy, val_accuracy)
    raise NotImplementedError


def extract_words(input_string) :
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits :
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts) :
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    dictionary = {} # maps word to unique index
    with open('stopwords.txt') as file:
        contents = file.read()
        contents = contents.replace("\n", " ").split()

    for text in texts :
        word_list = extract_words(text)

        for word in word_list :
            if word not in dictionary and word not in contents:
                dictionary[word] = len(dictionary)

    return dictionary

def extract_bow_feature_vectors(reviews, dictionary) :
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews) :
        word_list = extract_words(text)

        for word in word_list :
            if word in dictionary :
                feature_matrix[i, dictionary[word]] = word_list.count(word)

    return feature_matrix


def accuracy(preds, targets) :
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
