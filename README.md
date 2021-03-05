# Learning-Machine_Learning
---
0. Appendix. Mathematical review
    1. Linear Algebra
    2. Optimization Theory
    3. Probability Theory
    4. Information Theory
1. Artificial Neurons and Neural Networks
    1. Artificial Neurons
        1. Aritificial Neuron
        2. Components
        3. Compact Representation
        4. Signal-Flow graph representation
    2. Activation functions
        1. Threshold function(McCulloch and Pitts)
        2. Sigmoid function(Logistic funtion)
        3. Recified Linear Unit(ReLU)
        4. Softplus function
        5. Hyperbolic tangent function
        6. Leaky ReLU
        7. Maxout function
        8. ELU Function
        9. Properties of activation functions
    3. Stochastic artificial neuron
        1. Probablity of state transition
        2. Logistic Probability Model
    4. Neural Network Architectures
        1. Single-Layer Network
        2. Multilayer Feedforward Neural Network(FNN)
        3. Convolutional Neural Network(CNN)
        4. Recurrent Neural Network(RNN)
        5. Combined with a Pre-Processor
2. Rosenblatt's Perceptron
    1. Rosenblatt's Perceptron Model
        1. Overview
        2. Activation function : Sign function(threshold function)
        3. Network Architecture
        4. Assumption
        5. Training Problem Definition
        6. Training Algorithm for Perceptron
        7. Geometric Interpretation
    2. Perceptron Convergence Theorem
        1. Perceptron Convergence Theorem 
3. Regression
    1. Regressive and approximated models
        1. General regressive model
    2. Linear Regression
        1. Linearly approximated model
        2. Hypothesis
        3. Linear regression problem
        4. Learning algorithm : A numerical approach
        5. Learning algorithm : Least squares(One-shot learning approach)
        6. Recursive least squares
        7. Regularized least squares
        8. Comparisons
        9. Linear regression with basis functions
        10. Proper step size
        11. Good training samples
    3. Bayesian Regression
        1. Overview
        2. Maximum A Posteriori(MAP) estimation
        3. Maximum Likelihood(ML) estimation
        4. Bayesian linear regression with ML estimation
        5. Bayesian linear regression with MAP estimation
    4. Logistic and Softmax regression
        1. Logistic regression : hypothesis
        2. Logistic regression : learning based on gradient ascent algorithm
        3. Logistic regression : learning via Iterative Reweighted Least Squares(IRLS) based on Newton-Rapson method
        4. Logistic regression : Binary classification
        5. Softmax regression : Overview
        6. Softmax regression : Hypothesis
        7. Softmax regression : Derivative of softmax function
        8. Softmax regression : learning based on gradient ascent algorithm
        9. Softmax regression : learning via Iterative Reweighted Least Squares(IRLS) based on Newton-Rapson method
        10. Softmax regression : Multi-Class classification via softmax regression
    5. k-Nearest Neighbors(k-NN) Regression
        1. 𝑘-NN regression
        2.  𝑘-NN classfication
4. Statistical learning
    1. Wiener filter(Optimal linear MMSE filter)
        1. Overview
        2. Optimal linaer filtering problem
        3. Wiener filter(Limiting form of the LS solution)
    2. Steepest Gradient Descent Method and Least Mean Square Algorithm
        1. Gradient descent algorithm
        2. Two approaches for gradient descent
    3. Minimum Mean Square Error(MMSE) Estimator
    4. Review
5. Classification
    1. Definition of classification problem
    2. Linear Models for Classfication
        1. Linear discriminant for two classes
        2. Linear discriminant for multiple classes
        3. Linear models for classification
            1. Linear model for classification : Least squares for classification
            2. Linear model for classification : Fisher's linear discriminant
            3. Linear model for classification : Perceptron
    3. Probabilistic Approaches for Classification
        1. Statistics vs Bayesian Classification
        2. Probabilities in classification
        3. A simple binary classification
        4. Receiver Operating Characteristics (ROC)
        5. Bayesian classification : Minimum Bayes Risk Classifier for two classes
         6. Minimum Error Probability Classifier for two classes
        7. Bayesian classification : Minimum Bayes Risk Classifier for multiple classes
        8. Minimum Error Probability Classifier for multiple classes
        9. Naive Bayes classifier
        10. Assumptions of Naive Bayes classifier
        11. Bayes Gaussian Classifier
        12. Generative and discriminative approach
        13. Probabilistic Generative Models for two classes classification
        14. Probabilistic Generative Models for two classes classification with continuous features
        15. Probabilistic Generative Models for multiple classes classification
        16. Probabilistic Generative Models for multiple classes classification with continuous features
        17. Probabilistic discriminative models
6. Practical Issues in Machine Learning
    1. Bias-Variance tradeoff
        1. Bias-Variance decomposition of the MSE
    2. Generalization
        1. Overview
        2. Training and test data sets
    3. Overfitting
        1. How to avoid overfitting? 
            1. More training data
            2. Reducing the number of features(e.g., by PCA)
            3. Regularization
            4. Dropout
            5. Early-stopping
            6. Proper model selection
    4. Model selection
        1. Model selection with validation
            1. Model selection with validation set
            2. Multifold(K-fold) Cross-Validation
            3. Bootstrap Model Selection
        2. Model selection with criteria
            1. Akaike Information Criterion(AIC)
            2. Minimum Description Length(MDL) Criterion
    5. Curse of dimensionality 

7. Multilayer perceptron

8. Support Vector Machine

9. Restricted Bolzmamn Machines

10. Unsupervised learning

