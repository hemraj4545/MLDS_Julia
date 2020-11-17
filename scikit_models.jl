using ScikitLearn, MLDatasets
@sk_import linear_model: LogisticRegression
@sk_import model_selection: train_test_split
@sk_import metrics: accuracy_score
@sk_import tree:DecisionTreeClassifier
@sk_import ensemble: RandomForestClassifier

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
print("Accuracy for Logistic Regression: ",string(round(accuracy*100),3))

# Decision Tree Classifier
DTC = DecisionTreeClassifier()

fit!(DTC, features, labels)
prediction = predict(DTC, features)

accuracy = accuracy_score(prediction, labels)
print("Accuracy for Decision Tree Classifier: ",string(round(accuracy*100),3))

# Random Forest Classifier
RFC = RandomForestClassifier()

fit!(RFC, features, labels)
prediction = predict(RFC, features)

accuracy = accuracy_score(prediction, labels)
print("Accuracy for Random Forest Classifier: ",string(round(accuracy*100),3))
