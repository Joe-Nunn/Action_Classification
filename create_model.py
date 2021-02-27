"""
Trains and saves a model to classify human actions in sports

Author: Joe Nunn
"""

from actionclassifier.classifier import HumanActionClassifierTrainer

# Get path for directory containing images to train with
samples_folder_path = input("Please enter the path of the directory with images to train with: ")

# Get name of csv file containing the names of images and their corresponding labels
annotations_file = input("Please enter the name of the csv file containing image names and their labels: ")
annotations_file += ".csv"

# Get the same to save the model as
model_name = input("Please enter the name to save the model as: ")

print()

# Train model
trainer = HumanActionClassifierTrainer(samples_folder_path, annotations_file)
trainer.train(test=True, print_results=True)

# Save model
trainer.save_model(model_name)

# To stop program closing before output can be read
print()
input("Press enter to exit")
