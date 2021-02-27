"""
Module contains classes HumanActionClassifier and HumanActionClassifierTrainer
HumanActionClassifier can be used to load a pre-existing model and infer labels for images in a directory showing
actions in sport.
HumanActionClassifierTrainer can be used to train and export a model. Once a model is trained it may also be used for
inference.

Author: Joe Nunn
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from actionclassifier.dataset import TrainingDataset, InferenceDataset
from actionclassifier.network import NeuralNetwork


class HumanActionClassifier:
    """
    Classifies images of human actions in sports
    """
    NUM_CLASSES = 8
    CLASS_NAMES = ["walking", "running", "standing", "squatting", "pointing", "jostling", "batting", "throwing"]

    def __init__(self, batch_size=128, load_model_path=None):
        """
        initialises batch_size
        Creates neural network, either from a file or as a new untrained model
        Determines device samples will be evaluated on. CPU or graphics card if a CUDA enabled Nvidia card is available.

        :param batch_size: number of images to be evaluated at once
        :param load_model_path: path to load model. If no model path is specified a new untrained model will be created
        """
        if load_model_path is None:
            self.network = NeuralNetwork(self.NUM_CLASSES)
        else:
            self.network = torch.jit.load(load_model_path)
        self.batch_size = batch_size

        # Use Nvidia graphics card if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def get_label_names(self, labels):
        """
        Gets string of label names using a numerical list description of the labels
        :param labels: list of integer 0's and 1's. 0 indicating fit to label, 0 indicating no fit
        :return: string of comma and space separated label names
        """
        names = ""
        for i in range(self.NUM_CLASSES):
            if bool(labels[i]):
                names += self.CLASS_NAMES[i] + ", "
        return names[:-2]  # Return label names without excess comma and space

    def choose_labels(self, image_labels):
        """
        Chooses an image's labels by its label probabilities
        :param image_labels: a list of a list containing the chance that the image fits a label
        :return:  List of list. For each image its predicted labels. 0 as false 1 as true fit for label.
        """
        # Choose labels for each sample
        for label_probabilities in image_labels:
            # First 4 classes are mutually exclusive so most likely of them is chosen to be label
            largest_probability = -1
            for i in range(4):
                if label_probabilities[i] > largest_probability:
                    largest_probability = label_probabilities[i]
                    most_likely_label = i
            for i in range(4):
                if i == most_likely_label:
                    label_probabilities[i] = 1
                else:
                    label_probabilities[i] = 0
            # Remaining classes are chosen as being a label if over 50% probable
            for i in range(self.NUM_CLASSES):
                label_probabilities[i] = round(label_probabilities[i])

        return image_labels

    def infer_image_directory(self, images_folder):
        """
        Evaluates png images in the specified folder. Determines the labels for these images

        :param images_folder: path for the folder containing the images
        :return: list of strings containing the names of images and their labels
        """
        # Load images into dataset
        dataset = InferenceDataset(images_folder)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # Move model to GPU if using it
        self.network.to(self.device)
        self.network.eval()
        image_labels = []
        # Determine labels for images batch by batch
        for batch, data in enumerate(loader):
            images, filenames = data
            # Move batch of images to gpu if using it
            images = images.to(self.device)
            # Evaluate images in model
            predictions = self.network(images)
            # Get corresponding labels to output from the model
            predictions = predictions.tolist()
            labels = self.choose_labels(predictions)
            # Print labels for files
            for i in range(len(filenames)):
                image_labels.append(filenames[i] + ": " + self.get_label_names(labels[i]))
        return image_labels


class HumanActionClassifierTrainer(HumanActionClassifier):
    """
    Trains and tests a neural network to classify images of human actions in sports
    """

    def __init__(self, samples_folder_path, csv_name, batch_size=128, training_size=0.8):
        """
        initialises batch_size, training_size, and learning_rate to values set in parameters.
        Creates neural network.
        Loads sample images and labels, splitting them into a training and testing set.
        Determines device samples will be evaluated on. CPU or graphics card if a CUDA enabled Nvidia card is available.

        :param samples_folder_path: path to the folder containing samples and csv file
        :param csv_name: name of the csv file describing the name of the image files and their corresponding labels
        :param batch_size: number of samples to be evaluated at once
        :param training_size: decimal proportion of samples to be used for training in comparison to testing
        """
        super().__init__(batch_size)
        self.training_size = training_size
        # Load images
        self.dataset, self.train_loader, self.test_loader = self.prepare_data(samples_folder_path, csv_name)

    def train(self, test=True, print_results=False, learning_rate=0.0003, weight_decay=0.00001, epochs=13):
        """
        Trains the neural network using the train set of samples.
        Uses binary-cross entropy loss and Adam optimisation.

        Prints epoch and test data per epoch if print_results is true.

        :param: test: whether to test the model while training or not
        :param: learning_rate: learning rate used by the optimiser
        :param: weight_decay: weight decay used by the optimiser
        :param: epochs: number of epochs to train for
        :param: print_results: whether results of tests should be printed
        :return: tuple of two lists containing test results of the test set and train set. Lists empty if test false.
            Each entry in the list is for an epoch and is a tuple containing the following test results: f-score,
            precision, recall, true positives, false positives, false negatives, true positives by class (list),
            false positives by class (list), and false negatives by class (list).
        """
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.network.to(self.device)  # Move model onto GPU if available
        self.network.train()

        # Initialise lists to store test results in
        test_set_results = []
        train_set_results = []

        # Train the network with the training set samples epochs times
        for epoch in range(epochs):
            if print_results:
                print("epoch: " + str(epoch + 1) + "/" + str(epochs))
            running_loss = 0
            # Train the network one batch of samples at a time
            for batch, data in enumerate(self.train_loader):
                images, labels = data
                # Move batch of images and labels to graphics card if using it
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward batch through network
                outputs = self.network(images)
                # Calculate how wrong the network output is
                loss = criterion(outputs, labels)
                # Adjust weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if test:
                if print_results:
                    # Print loss
                    print("Loss: " + str(running_loss / len(self.train_loader)))

                # Test model on test samples
                if print_results:
                    print("test set:")
                test_set_results.append(self.test(self.test_loader, print_results))
                if print_results:
                    print("train set:")
                train_set_results.append(self.test(self.train_loader, print_results))
                if print_results:
                    print()
        return test_set_results, train_set_results

    def test(self, loader=None, print_results=False):
        """
        Tests the accuracy of the neural network using test sample set.

        Calculates the f-score, precision, and recall scores of the network.
        Calculates the number of true positives, false positives and false negatives predicted by the network overall and
        by class.

        :param: loader containing samples to test network with. Default is attribute test_loader
        :param: print_results: whether the results of the test should be printed
        :return: tuple of test results: f-score, precision, recall, true positives, false positives, false negatives,
            true positives by class, false positives by class, false negatives by class
        """
        # Set the loader to the default (test_loader) if no loader is specified in parameters
        if loader is None:
            loader = self.test_loader

        # Initialise values for true positives, false negatives, and false positives
        class_true_positives = [0] * self.NUM_CLASSES
        class_false_positive = [0] * self.NUM_CLASSES
        class_false_negative = [0] * self.NUM_CLASSES
        total_true_positives = 0
        total_false_positive = 0
        total_false_negative = 0

        self.network.eval()
        self.network.to(self.device)  # Move model to graphics card if using it

        with torch.no_grad():  # gradient isn't needed in training turning it off increases performance
            # Test the network one batch of samples at a time
            for batch_num, data in enumerate(loader):
                images, labels = data
                # Move batch of images to graphics card if using it
                images = images.to(self.device)
                # Forward batch through network
                predictions = self.network(images)
                # Check correctness of label predictions for the batch
                predictions = predictions.tolist()
                labels = labels.tolist()
                predictions = self.choose_labels(predictions)
                for i in range(len(predictions)):
                    for j in range(self.NUM_CLASSES):
                        # Update prediction accuracy statistics
                        label_prediction = predictions[i][j]
                        if label_prediction == labels[i][j]:
                            if bool(label_prediction):
                                class_true_positives[j] += 1
                                total_true_positives += 1
                        else:
                            if bool(label_prediction):
                                class_false_positive[j] += 1
                                total_false_positive += 1
                            else:
                                class_false_negative[j] += 1
                                total_false_negative += 1

        # Calculate accuracy scores
        total_precision = self.calc_precision(total_true_positives, total_false_positive)
        total_recall = self.calc_recall(total_true_positives, total_false_negative)
        total_f_score = self.calc_f_score(total_precision, total_recall)

        if print_results:
            # Print statistics about accuracy of predictions
            print("F-score: " + "{:.2f}".format(total_f_score))
            print("Precision: " + "{:.2f}".format(total_precision))
            print("Recall: " + "{:.2f}".format(total_recall))
            print("True positives: " + str(total_true_positives))
            print("False positives: " + str(total_false_positive))
            print("False negatives: " + str(total_false_negative))
            print("True positives per class: " + str(class_true_positives))
            print("False positives per class: " + str(class_false_positive))
            print("False negatives per class: " + str(class_false_negative))

        return total_f_score, total_recall, total_recall, total_true_positives, total_false_positive, \
            total_false_negative, class_true_positives, class_false_positive, class_false_negative

    @staticmethod
    def calc_precision(true_positive, false_positive):
        """
        Calculates the precision of predictions
        :param true_positive: how many labels were correctly predicted as true
        :param false_positive: how many labels were incorrectly predicted as true
        :return: decimal of precision
        """
        if true_positive + false_positive > 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0

    @staticmethod
    def calc_recall(true_positive, false_negative):
        """
        Calculates the recall of predictions
        :param true_positive: how many labels were correctly predicted as true
        :param false_negative: how many labels were incorrectly predicted as false
        :return: decimal of recall
        """
        if true_positive + false_negative > 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0

    @staticmethod
    def calc_f_score(precision, recall):
        """
        Calculates the f-score of predictions
        :param precision: how precise the label predictions are
        :param recall: the recall rate of the label predictions
        :return: decimal of f-score
        """
        if precision + recall > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0

    def prepare_data(self, folder_path, csv_name):
        """
        Load labels and samples.
        Split them into test and train loaders.

        :param folder_path: path to folder containing images and csv file
        :param csv_name: name of csv file naming image names and corresponding labels

        :return: full dataset, loader for training samples, loader for testing samples
        """
        # Load samples
        dataset = TrainingDataset(csv_file_name=csv_name, directory_name=folder_path)
        # Calculate number of samples to be used for training and testing using training_size
        n_train_images = round(dataset.__len__() * self.training_size)
        n_test_images = dataset.__len__() - n_train_images
        # Split the samples into training and testing set.
        train_set, test_set = torch.utils.data.random_split(dataset, [n_train_images, n_test_images])
        train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)
        return dataset, train_loader, test_loader

    def save_model(self, name):
        """
        Saves model as a TorchScript model

        :param name: name to save the model as
        """
        self.network.eval()
        self.network.to("cpu")
        example_input = torch.rand(1, 3, 128, 128)
        traced = torch.jit.trace(self.network, example_inputs=example_input)
        torch.jit.save(traced, name + ".pt")
        print("Model saved as " + name + ".pt")
