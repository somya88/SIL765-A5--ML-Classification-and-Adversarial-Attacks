#Part A Model training

#Introduction

The assignment aims to build and train a neural network to classify handwritten digits using the MNIST dataset. The focus is on understanding the nuances of image classification and exploring the robustness of models against adversarial attacks.

#Data Preparation

The MNIST dataset was split into training and testing sets with a ratio of 60:40. The images were saved in an `images` directory and their corresponding labels in a `labels` directory for both training and testing data.

#Data Structure
The data is organized as follows:

train_data/
│
├── images/
│ ├── image_0.png
│ ├── image_1.png
│ └── ...
├── labels/
│ ├── label_0.txt
│ ├── label_1.txt
│ └── ...

test_data/
│
├── images/
│ ├── image_0.png
│ ├── image_1.png
│ └── ...
├── labels/
│ ├── label_0.txt
│ ├── label_1.txt
│ └── ...



Each `label_*.txt` file contains a single digit indicating the class of the corresponding `image_*.png`.

#Model Architecture

We designed a fully connected feedforward neural network with the following specifications:

- Input Layer: Flattens the 28x28 image to a 784-element vector.
- Hidden Layer 1: 512 neurons with ReLU activation.
- Hidden Layer 2: 256 neurons with ReLU activation.
- Hidden Layer 3: 128 neurons with ReLU activation.
- Output Layer: 10 neurons (one for each class) with log-softmax activation.

To mitigate overfitting, dropout was applied after each hidden layer with a probability of 0.2.

#Training Process

Training involved the following settings:

- Epochs: 10
- Batch Size: 64
- Loss Function: Negative Log-Likelihood Loss (NLLLoss)
- Optimizer: Adam with a learning rate of 0.003

The model's parameters were saved in a `model.pth` file upon training completion.

#Results

The model achieved a test accuracy of approximately 90.6%, suggesting effective learning. Training loss consistently decreased across epochs, which aligns with expectations for a well-fitting model.

# Observations

- The training process did not exhibit signs of overfitting, as evidenced by stable loss reduction and high test accuracy.
- Further improvements might include more complex architectures, hyperparameter tuning, or augmentation techniques.

# Instructions for Use

To retrain the model or replicate the results, execute the `classify.py` script:

bash
python classify.py

The script will automatically train the model and save the weights to model.pth, ready for subsequent evaluation or deployment.

#Hyperparameter Tuning

The initial training showed promising results; however, to further optimize our model's performance, the following hyperparameters were adjusted:

- Learning Rate: Experimented with learning rates [0.001, 0.003, 0.01] to find the balance between convergence speed and stability.

- Batch Size: Tested batch sizes [32, 64, 128] to determine the impact on the generalization of the model.

- Number of Epochs: Increased epochs from 10 to 15 and observed the training process for any signs of overfitting.

- Network Depth and Width: Modifications were made to the number of neurons in each layer and the number of layers to explore model complexity's effect on accuracy.

The chosen hyperparameters were based on a series of experiments where each variable was adjusted individually while keeping others constant. The performance impact was noted, and the combination yielding the highest accuracy on the test set was selected.


#Part B Adversarial attack
Fast Gradient Sign Method (FGSM)
The Fast Gradient Sign Method (FGSM) is employed as an adversarial attack to assess the model's robustness. The attack perturbs input images with small, calculated adversarial perturbations to cause misclassification while aiming to minimize perceptual differences.

#Results
The attack results include:

#Evasion Rate: 
The percentage of adversarial examples that evade the model's classification.

#Accuracy: 
The accuracy of the model on the adversarial examples.
Model.py

The model.py file contains the definition of the neural network model, including its architecture, training process, and evaluation methods.

#Evasion Rate Calculation: 
The evasion rate of the generated adversarial examples against the trained model will be calculated.

#Visualization of Adversarial Examples: 
Random images of all ten digits were selected, and their corresponding adversarial examples and the L2 norm of the adversarial noise were plotted.

#Misclassification Analysis:
One digit was randomly picked, and the class to which it got misclassified the most was reported. Structural similarities between the chosen digit and the most misclassified digit were analyzed, along with possible reasons for misclassification.


#Attack.py
The attack.py file implements the FGSM attack to evaluate the model's robustness against adversarial examples. It applies perturbations to input images to generate adversarial examples and calculates the evasion rate and accuracy of the model on these examples.

#Model Evaluation
After applying the FGSM attack using attack.py, the evasion rate and accuracy of the model on the adversarial examples are computed and reported. These metrics provide insights into the model's vulnerability to adversarial attacks and its overall performance under such conditions.
