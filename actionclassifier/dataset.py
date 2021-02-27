"""
Module contains classes TrainingDataset and InferenceDataset
Both classes are used to load images.
TrainingDataset can be used to load and access images and labels described in a csv file.
InferenceDataset can be used to load and access images and their filenames contained in a directory.

Author: Joe Nunn
"""

import torch
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io


class TrainingDataset(Dataset):
    """
    Dataset to load and store images for use in training
    """
    def __init__(self, csv_file_name, directory_name):
        """
        Loads image names and labels of samples from csv file.
        Initialises transformation, and directory name.

        :param csv_file_name: name of csv file containing names of the images and their corresponding labels
        :param directory_name: path to the folder which contains the images and csv file
        """
        self.root_dir = directory_name
        CSV_path = os.path.join(self.root_dir, csv_file_name)
        self.annotations = pd.read_csv(CSV_path)

    def __len__(self):
        """
        Gets the number of entries in the data set

        :return: number of entries in the data set
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Gets an image and its labels at index in form of tensor

        :return image and labels tensor
        """
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        labels = torch.FloatTensor(self.annotations.iloc[index, 1:])  # tuple of labels
        transform = transforms.ToTensor()
        image = transform(image)
        image = image.float()

        return image, labels


class InferenceDataset(Dataset):
    def __init__(self, images_folder):
        """
        Loads images and their names for use in inference

        :param: images_folder: path for the folder containing the images to load
        """
        self.images = []
        self.filenames = []
        # Every file with extension .png added to dataset
        for filename in os.listdir(images_folder):
            if ".png" in filename:
                img_path = os.path.join(images_folder, filename)
                image = io.imread(img_path)
                self.images.append(image)
                self.filenames.append(filename)

    def __len__(self):
        """
        Gets the number of images in the data set

        :return: number of images in the data set
        """
        return len(self.images)

    def __getitem__(self, i):
        """
        Gets filename and image at index i
        Transforms image into tensor

        :param i: index of item to get
        :return: tuple of filename and image tensor
        """
        transform = transforms.ToTensor()
        image = transform(self.images[i])
        return image, self.filenames[i]
