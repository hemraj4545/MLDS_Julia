using ScikitLearn, MLDatasets
@sk_import linear_model: LogisticRegression
@sk_import model_selection: train_test_split
@sk_import metrics: accuracy_score
@sk_import tree:DecisionTreeClassifier
@sk_import ensemble: RandomForestClassifier
@sk_import svm:SVC
@sk_import naive_bayes: (GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB)

# Loading the Iris Dataset from MLDatasets.jl
features = Iris.features()
labels = Iris.labels()
# Transpose the features table
features = features'

# Splitting the dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=5)

# Logistic Regression Model
LR = LogisticRegression(fit_intercept=true)

fit!(LR, x_train, y_train)
prediction = predict(LR, x_test)

accuracy = accuracy_score(prediction, y_test)
println("Accuracy for Logistic Regression : ",string(round(accuracy*100),3))

# Decision Tree Classifier
DTC = DecisionTreeClassifier()

fit!(DTC, x_train, y_train)
prediction = predict(DTC, x_test)

accuracy = accuracy_score(prediction, y_test)
println("Accuracy for Decision Tree Classifier : ",string(round(accuracy*100),3))

# Random Forest Classifier
RFC = RandomForestClassifier()

fit!(RFC, x_train, y_train)
prediction = predict(RFC, x_test)

accuracy = accuracy_score(prediction, y_test)
println("Accuracy for Random Forest Classifier: ",string(round(accuracy*100),3))

# Support Vector Machine
SVM = SVC()

fit!(SVM, features, labels)
prediction = predict(SVM, features)

accuracy = accuracy_score(prediction, labels)
println("Accuracy for Support Vector Machine: ",string(round(accuracy*100),3))

# Naive-Bayes Models
GNB = GaussianNB()
BNB = BernoulliNB()
CNB = CategoricalNB() 
CcNB = ComplementNB()
MNB = MultinomialNB()

# Gaussian Naive Bayes
fit!(GNB, features, labels)
prediction = predict(GNB, features)

accuracy = accuracy_score(prediction, labels)
println("Accuracy for Gaussian Naive Bayes: ",string(round(accuracy*100),3))

# Bernoulli Naive Bayes
fit!(BNB, features, labels)
prediction = predict(BNB, features)

accuracy = accuracy_score(prediction, labels)
println("Accuracy for Bernoulli Naive Bayes: ",string(round(accuracy*100),3))

# Categorical Naive Bayes
fit!(CNB, features, labels)
prediction = predict(CNB, features)

accuracy = accuracy_score(prediction, labels)
println("Accuracy for Categorical Naive Bayes: ",string(round(accuracy*100),3))

# Complement Naive Bayes
fit!(CcNB, features, labels)
prediction = predict(CcNB, features)

accuracy = accuracy_score(prediction, labels)
println("Accuracy for Complement Naive Bayes: ",string(round(accuracy*100),3))

# Multinomial Naive Bayes
fit!(MNB, features, labels)
prediction = predict(MNB, features)

accuracy = accuracy_score(prediction, labels)
println("Accuracy for Multinomial Naive Bayes: ",string(round(accuracy*100),3))
