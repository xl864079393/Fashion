In this project, we classify images from the Fashion-MNIST dataset using k-Nearest Neighbors (KNN), logistic regression, feed-forward neural networks, and convolutional neural networks (CNN). Through comparative analysis, we discovered that feed-forward neural networks is the best model for this dataset, achieving a test accuracy of 90.06%.
Data Description
The dataset consists of 70,000 28x28 color images that are evenly distributed across 10 classes such as T-shirt, Trouser, Pullover, Dress, CoatSandal, Shirt, Sneaker, Bag, and Ankle boot. There are 70,000 images per class with 60,000 training and 10,000 testing images per class. To explore the dataset, we visualized 21 random datasets with their labels.

The research paper titled "CNN Model for Image Classification on MNIST and Fashion-MNIST Dataset" by Shivam S. et al. indicates that CNNs, characterized by their convolutional, pooling, and fully connected layers, have emerged as a powerful architecture in deep learning, particularly for tasks such as image classification. Fashion-MNIST is also a dataset commonly used in machine learning research for image classification. The authors demonstrate how CNN works effectively by comparing it with MNIST in their paper. Therefore, in this report, we are testing three different classifiers and CNN to show the accuracy of each model on the Fashion-MNIST Dataset.
Classifiers
Except for logistic regression, we all used PyTorch software because it effectively supports GPU acceleration, enabling us to flexibly modify the code.
KNN classifier: Its essence lies in identifying the K nearest samples to the one being evaluated within the training dataset based on distance metrics. For classification, it tallies the categories of these nearest neighbors and assigns the most frequent category to the target sample. Meanwhile, in regression prediction, it computes the average value of these K neighbors to estimate the target value. The hyperparameter only includes K [1,2,3,4,5,6,7,8,9,10, 100, 500, 1000], which dictates the count of samples from the training dataset that are closest to the samples under classification, with the category possessing the highest frequency deemed as the category for the samples being classified.
Logistic Regression: This is a binary classification algorithm that uses a sigmoid function to map input features to a probability score between 0 and 1, indicating the likelihood of an instance belonging to a specific class. The hyperparameters of logistic regression we trained are regularization strength(C), which ranges [0.001, 0.005, 0.01, 0.05, 10], penalty, solver, and maximum iteration. The software we used is sklearn.linear_model.LogisticRegression.
Feedforward Neural Network: This type of model is called Feedforward because input flows through functions, passes intermediate computation used to define f, and eventually reaches the output y. The model architecture features a flatten layer to convert 2D image tensors into 1D vectors, followed by several fully-connected layers with batch normalization and dropout techniques. In our grid search optimization, we explored learning rates [0.001, 0.0005, 0.0001] and dropout rates [0.25, 0.5, 0.75] for the model to enhance validation accuracy. By balancing the learning rate for efficient weight updates and the dropout rate to mitigate overfitting, we identified the optimal hyperparameter combination. 
Convolutional Neural Networks: CNNs is a classifier that uses filters to extract features and max-pooling to downsample, while non-linear activation functions like ReLU introduce complexity for learning patterns and making accurate predictions, especially in tasks like image recognition. The hyperparameters of the Convolutional Neural Networks we trained are as follows: learning rates of [0.0005, 0.001, 0.005, 0.01], varying numbers of convolutional layers from 1 to 4, filter sizes of [3, 5, 7, 9], and numbers of filters specified as [[16, 32, 32], [16, 32, 64], [32, 32, 64], [32, 64, 128]] for each set of three convolutional layers. 
Experimental Setup
In this report, we mainly examined the classification accuracy. Since Fashion-MNIST already has 60,000 training and 10,000 testing images per class, we will partition the training data into 50,000 training and 10,000 validation sets, which is approximately 71.43% training set, 14.29% validation set, and 14.29% testing set. To ensure reproducibility of the experiments, the random_state is set to 1234. If there is missing data in the training set, we drop the rows with any missingness.
K-Nearest Neighbors :
In this context, the majority of distance calculations are performed using matrix computations. 
Here are two versions of the model, one using sklearn and the other using PyTorch. The main purpose is to explore PyTorch's optimization for matrix calculations, with time being a crucial metric as well. Additionally, another crucial aspect is accuracy. While KNN demonstrates impressive accuracy on MNIST data, its performance falters when confronted with intricate image patterns, notably evident in Fashion MNIST. This challenge stems from inherent limitations within the KNN algorithm. Mere adjustments to the K parameter fail to adequately surmount these constraints. Thus, in this investigation, We integrate the Histogram of Oriented Gradients (HOG) technique to extract pivotal image features for recognition. This strategic augmentation serves to effectively alleviate potential underfitting in KNN, resulting in a substantial enhancement of its accuracy. 

As shown in the figure, HOG highlights the gradients within the images, enabling the KNN algorithm to better differentiate distances between images. This amplifies the distances between various images, facilitating improved classification accuracy. Due to the simplicity of the KNN algorithm, data is typically divided into two main subsets: the training set and the testing set. The absence of a validation set is justified by the fact that it offers minimal assistance in optimizing hyperparameters or improving accuracy in KNN. Regarding the selection of hyperparameters, We opted to vary the value of K to observe changes in the model. K = [1,2,3,4,5,6,7,8,9,10, 100, 500, 1000].
Logistic regression:
While initially testing each possible combination of parameters randomly, we chose the model with best training accuracy. Subsequently, with this configuration, we proceeded to evaluate various regularization strengths(C) [0.001, 0.005, 0.01, 0.05, 10] since it plays a significant role in controlling model complexity, preventing overfitting, and improving generalization performance.
For each c in C:
	Create the regression model with above configuration but with 600 maximum iteration and C=c
	Fit the model with X_train and y_train
Using X_val to predict y_val and calculate the validation accuracy using the accuracy_score function
Feedforward Neural Network:
Before tuning learning rate init for the Feedforward Neural Network classifier, we performed grid search with a selected range of hyperparameters to obtain the preliminary setup. After grid search, the optimal learning rate and dropout settings were applied to finalize the model, which was then assessed on a test set to determine generalization ability. The learning rate controls the speed at which the model updates its weights, while the dropout rate helps prevent overfitting by randomly omitting a proportion of neuron connections during training. Our model, defined with a sequence of linear, batch normalization, ReLU activation, and dropout layers, was trained and evaluated using PyTorch. Our training function computed losses and accuracies for each epoch, adjusting weights with an Adam optimizer and CrossEntropyLoss. 
Convolutional Neural Network:
The CNN architecture was designed with convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification. After training, the model was evaluated on the test dataset using classification accuracy as the primary metric, with consideration given to additional metrics like precision and recall for multiclass classification evaluation. Fine-tuning and optimization were conducted iteratively to enhance model performance. The choice of learning rate(including [0.0005, 0.001, 0.005, 0.01]) may lead to faster convergence but risk overshooting optimal solutions. The number of epochs(10 and 50) influences how long the model trains for, potentially affecting both convergence and overfitting. The different Convolutional Layers (1 to 4) make models to capture different complex patterns in the data, potentially affecting accuracy. The filter size [3,5,7,9] captures different ranges of patterns and may also increase computational complexity and the risk of overfitting.
Experimental Results
While validation accuracy served as the main criterion during model training, the table below presents the test accuracy results obtained from the trained models. The Feedforward Neural Network demonstrates the highest test accuracy for Fashion-MNIST, while KNN and CNN achieve a comparable accuracy of about 89.75%. In contrast, logistic regression only achieves 84.43%, suggesting its limitations for image classification.
Model
Parameters that are tuned
Test Accuracy
Other Results
KNN
K = 4
0.897





See Appendix
Logistic Regression
C=0.005
0.8443
Feedforward Neural Network
Learning rate: 0.0001
Dropout: 0.25
0.9006
Convolutional Neural Networks
Number of Layers: 3
Number of Filters 32,32,64
Kernel Size: 5 
Pooling Size: 2x2
0.8984

Insights
One primary insight from this project was the understanding of the challenges associated with classifying visually similar categories. The performance of feed-forward neural networks, achieving a test accuracy of 90.06%, highlights the significant potential of well-tuned neural network architectures in handling complex image classification tasks. 
 One of the primary strengths of using machine learning models for image classification, as demonstrated in this project, is their ability to learn and generalize from data without explicit programming for each classification category, they just learn from data and make predictions across a broad range of categories. This capability is particularly valuable in applications such as automated inventory management. Speculating on real-world applications, one could use public cameras in legally permissible public spaces, such as airports, to help the police detect criminals' faces and identities. This could involve advanced models that better capture the intricacies of visual data, or hybrid approaches that combine multiple types of data for more accurate classifications.
In short, this project not only provided valuable insights into the capabilities and limitations of different machine learning models for image classification but also highlighted the critical considerations for their application in real-world scenarios.
Contributions
For this assignment, we divided the workload into parts. Initially, each team member was assigned to train and test one model: Xiang Li trained K-Nearest Neighbors (KNN), Tzu Hsuan Huang trained Logistic Regression, and Douzi Ma trained a Feedforward Neural Network. After completing individual training sessions, we gathered our findings and engaged in a comprehensive discussion regarding the insights obtained from these three models. Subsequently, Tzu Hsuan trained and tested Convolutional Neural Networks (CNNs) and Xiang and Douzi concluded by compiling our findings into the final written report.






























Appendix
K-Nearest Neighbors :
In regular KNN, the average loss stands at 0.148, with an achievable accuracy of 0.867. However, there is considerable fluctuation in the choice of K values within the range of 1 to 10, showing no discernible pattern. 
Furthermore, as the K value continues to increase, the loss value also steadily rises, indicating a clear case of overfitting in the KNN model. 

In the KNN+HOG algorithm, most of the issues encountered in regular KNN are alleviated to a significant extent. The average loss in this algorithm stands at 0.112, with an achievable accuracy of 0.897. Compared to traditional KNN, the KNN+HOG algorithm notably enhances accuracy, underscoring the substantial benefits of preprocessing images to intensify and highlight edge gradient cells, thereby aiding the KNN algorithm significantly.

 Additionally, it is evident that under varying K values, the fluctuation in KNN+HOG is noticeably smaller compared to traditional KNN. Beyond K=4, the curve tends to stabilize, reflecting HOG's ability to address overfitting concerns. Furthermore, in the confusion matrix, it's apparent that each color block is noticeably brighter, indicative of improved classification accuracy. 
Another crucial parameter to consider is the runtime. Unfortunately, the runtime of KNN+HOG significantly exceeds that of traditional KNN. After evaluating 15 different K values, KNN has a runtime of 13 seconds, whereas KNN+HOG requires 380 seconds. The runtime of KNN+HOG is approximately 30 times longer than that of KNN.
Feedforward Neural Network:
The grid search was performed over a combination of learning rates (lr) and dropout rates, across 10 epochs for each combination. The learning rates tested were 0.001, 0.0005, and 0.0001, and the dropout rates were 0.25, 0.5, and 0.75.

The table below summarizes the validation accuracy achieved for each combination:
Learning rate
0.01
0.0005
0.0001
Dropout
0.25
0.5
0.75
0.25
0.5
0.75
0.25
0.5
0.75
accuracy
89.52%
88.66%
86.80%
89.25%
88.87%
86.68%
89.89%
88.76%
88.32%

Based on the table above, the combination of a 0.0001 learning rate and a 0.25 dropout rate yielded the highest validation accuracy of 89.89%, indicating the model's ability to learn effectively while minimizing overfitting.

Following this, the training accuracy climbs consistently across the epochs, starting from 81.898% and reaching 94.746% by epoch 30. Validation accuracy starts from 85.500% and peaking at 90.410% in the final epoch. Finally, the model with the optimal hyperparameters led to a test set accuracy of 90.06%.

According to the figure above, we can see that the training loss starts at a higher value and steadily decreases as the number of epochs increases, indicating that the model is effectively learning from the training dataset. While the validation loss trends similarly start higher and decrease as the model trains. However, in some configurations, the validation loss plateaus or even increases slightly, indicating the beginning of overfitting. The model starts to learn noise or patterns specific to the training set that do not generalize well to the validation set.
The trends of decreasing loss, both in training and validation phases, coupled with a small gap between them, underscore the model's effective learning and generalization capabilities under the chosen hyperparameters.
Convolutional Neural Network:
The convolutional neural network (CNN) using different Learning rate achieving validation set accuracies ranging from approximately 88.96% to 90.15% and training set accuracies ranging from around 91.868% to 93.194%. These results signify the model's capability to effectively learn and generalize patterns within the dataset. Notably, the model exhibits stable training progress, with consistent decreases in loss across epochs, indicative of successful learning. 

The learning curves for the CNN model trained on a subset of 10,000 samples from the training set over 50 epochs reveal a notable trend. The loss decreases from 0.7458 to 0.0110 by the 26th epoch in one instance, and from 0.7792 to 0.0089 by the 26th epoch in another. These reductions suggest that the model effectively learns to minimize prediction errors as training progresses. Furthermore, the accuracy metrics on both the validation and training sets consistently show high performance, with validation accuracies ranging from approximately 88.41% to 89.45% and training accuracies ranging from about 98.49% to 99.44%.

Also, the test reveals a trend where increasing the number of convolutional layers initially improves performance, with accuracies on the validation set rising from 88.37% for a single layer to 89.23% for three layers. However, further stacking of convolutional layers beyond the third one yields diminishing returns, as evidenced by the marginal improvement to 89.11% accuracy with four layers. This suggests an optimal depth for the CNN architecture, where adding more layers does not significantly enhance performance but may increase computational complexity and risk overfitting.

The difference size of filter particularly focuses on various filter sizes (3x3, 5x5, 7x7, and 9x9) across three convolutional layers. Each model comprises 16 filters per layer and undergoes training with a learning rate of 0.001, a batch size of 4, and for five epochs. The evaluation metrics reveal distinct patterns in model performance across different filter sizes. Notably, models with filter sizes of 5x5 and 7x7 consistently demonstrate slightly higher accuracies compared to those with 3x3 and 9x9 filters, achieving validation set accuracies ranging from 89.11% to 89.42% and training set accuracies ranging from 90.772% to 91.97%. The model utilizing a 7x7 filter size emerges as the top performer, showcasing the highest accuracies on both validation (89.27%) and training (92.302%) sets.

To train different number filters in CNN, We employ a kernel size of 5x5, a padding size of 2, and max pooling with a size of 2x2. The models were trained with a learning rate of 0.001, a batch size of 4, and for five epochs. The accuracy metrics showcase consistent performance improvements across epochs, with accuracies on both the validation and training sets gradually increasing over time. Specifically, the model achieves accuracies ranging from 90.09% to 90.15% on the validation set and from 93.25% to 94.25% on the training set.

Overall, the model architecture comprises three convolutional layers, with 32 filters in the first and second layers and 64 filters in the third layer. The convolutional layers utilize a kernel size of 5x5 and a padding size of 2. Max pooling with a size of 2x2 is employed. The model was trained with a learning rate of 0.001, a batch size of 4, and for five epochs. The accuracy of the model on the 10,000-test set is reported to be 89.84%. 
Logistic Regression: 
The logistic regression yields an accuracy ranging from 83% to 85%. The tested metrics mainly focus on the regularization strength, solver, and penalty. However, after exploring around 8 different parameter sets, the accuracy consistently falls within the range of 0.84 to 0.852. 
Using the test 6 model, 'lbfgs' solver with 'l2' penalty, configured with a maximum of 600 iterations to test different regularization strengths, the accuracy also ranges from 0.83 to 0.852. Furthermore, after reaching 0.005, regularization strengths lead to continued increases in training accuracy but decreases in validation accuracy, implying that the model is overfitting beyond that value. Therefore, selecting a regularization strength of 0.005 results in a superior model.


Parameter
Validation accuracy
C=0.001
0.8436
C=0.005
0.852
C=0.01
0.8513
C=0.05
0.8457
C=10
0.8351

