# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# NLTK setup
nltk.download('stopwords')  # Downloading stopwords

# Load the dataset
spam = pd.read_csv("spam.csv")

# Check for null values in the data
print(spam.isnull().sum())

# Check the first few records
print(spam.head())

# Rename the columns for clarity
spam = spam[['v1', 'v2']]
spam.columns = ['label', 'message']

# Check the size of the dataset
print(spam.shape)

# Check the distribution of classes
print(spam.groupby('label').size())

# Plotting the label distribution
spam['label'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Spam vs Not Spam Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Preprocessing the text data
ps = PorterStemmer()
corpus = []
for i in range(0, len(spam)):
    review = re.sub('[^a-zA-Z]', ' ', spam['message'][i])  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Split into words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]  # Remove stopwords and apply stemming
    review = ' '.join(review)  # Join the words back together
    corpus.append(review)

# Check the first 5 processed texts
print(corpus[:5])

# Creating the Bag of Words model
cv = CountVectorizer(max_features=4000)  # Limit to top 4000 words
X = cv.fit_transform(corpus).toarray()  # Convert the corpus into a matrix of token counts
Y = pd.get_dummies(spam['label'])  # Convert labels into 0 or 1
Y = Y.iloc[:, 1].values  # Only keep the 'spam' column (1 for spam, 0 for not spam)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Create and train the models
model1 = RandomForestClassifier()
model1.fit(X_train, Y_train)

model2 = DecisionTreeClassifier()
model2.fit(X_train, Y_train)

model3 = MultinomialNB()
model3.fit(X_train, Y_train)

# Make predictions using the models
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# Confusion matrix and accuracy for Random Forest
cm1 = confusion_matrix(Y_test, pred1)
print("Random Forest Confusion Matrix: \n", cm1)
print("Random Forest Accuracy: ", accuracy_score(Y_test, pred1))

# Confusion matrix and accuracy for Decision Tree
cm2 = confusion_matrix(Y_test, pred2)
print("Decision Tree Confusion Matrix: \n", cm2)
print("Decision Tree Accuracy: ", accuracy_score(Y_test, pred2))

# Confusion matrix and accuracy for Multinomial Naive Bayes
cm3 = confusion_matrix(Y_test, pred3)
print("Multinomial Naive Bayes Confusion Matrix: \n", cm3)
print("Multinomial Naive Bayes Accuracy: ", accuracy_score(Y_test, pred3))

# Visualize confusion matrices using heatmaps
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix - Multinomial Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot model accuracy comparison
accuracies = {
    'Random Forest': accuracy_score(Y_test, pred1),
    'Decision Tree': accuracy_score(Y_test, pred2),
    'Multinomial Naive Bayes': accuracy_score(Y_test, pred3)
}

plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'orange', 'green'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Classification reports for each model
report1 = classification_report(Y_test, pred1)
print("Classification Report for Random Forest: \n", report1)

report2 = classification_report(Y_test, pred2)
print("Classification Report for Decision Tree: \n", report2)

report3 = classification_report(Y_test, pred3)
print("Classification Report for Multinomial Naive Bayes: \n", report3)

# The best model is Multinomial Naive Bayes based on accuracy and performance
print("\nBest Model: Multinomial Naive Bayes")

# Saving models to disk
filename = "RFC.pkl"
pickle.dump(model1, open(filename, 'wb'))
filename = "DTC.pkl"
pickle.dump(model2, open(filename, 'wb'))
filename = "MNB.pkl"
pickle.dump(model3, open(filename, 'wb'))
pickle.dump(cv, open("cv.pkl", "wb"))  # Save the CountVectorizer
print("Saved all models.")

# Loading the saved model and CountVectorizer
model = pickle.load(open("MNB.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

# Preprocessing function to use during prediction
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

# Input email text
email_text = " Hello professor Hourani, I don’t have any updates on the project at this time. Could we cancel our meeting? Thank you"

# Preprocess the input email text
processed_email = preprocess_text(email_text)

# Convert the email to Bag of Words representation
email_vector = cv.transform([processed_email]).toarray()

# Make prediction
prediction = model.predict(email_vector)
print("Prediction: Spam" if prediction[0] == 1 else "Prediction: Not Spam")

