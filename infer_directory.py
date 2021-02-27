"""
Loads a model and uses it to infer labels for images in a directory showing human actions in sport

Author: Joe Nunn
"""

from action_classifier.classifier import HumanActionClassifier

# Create classifier using model specified by user
model_path = input("Please enter path of model (ending with .pt) to use: ")
classifier = HumanActionClassifier(load_model_path=model_path)
print()

# Get labels for images in directory specified by the user
image_directory_path = input("Please enter the path of the directory with images to infer: ")
print("Evaluating images...")
labels = classifier.infer_image_directory(image_directory_path)
print()

# Output the label for each image
for label in labels:
    print(label)

# To stop program closing before output can be read
print()
input("Press enter to exit")
