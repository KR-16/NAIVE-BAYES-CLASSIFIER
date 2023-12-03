import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Step 1: Data Loading and Preprocessing
data = pd.read_csv('train_essays.csv')  # Replace 'your_data.csv' with your actual dataset file

# Tokenize the essays
data['tokens'] = data['text'].apply(lambda x: word_tokenize(x.lower()))

# Split the dataset into train and development sets
train_data, dev_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 2: Build Vocabulary
all_tokens = [token for tokens in train_data['tokens'] for token in tokens]
token_counts = pd.Series(all_tokens).value_counts()

# Create a vocabulary list (omit rare words with occurrences less than 5)
vocabulary = list(token_counts[token_counts >= 5].index)

# Create a reverse index for the vocabulary
reverse_index = {word: idx for idx, word in enumerate(vocabulary)}

# Step 3: Naive Bayes Classifier Implementation
class NaiveBayesClassifier:
    def __init__(self, vocabulary, reverse_index):
        self.vocabulary = vocabulary
        self.reverse_index = reverse_index
        self.class_word_counts = {'0': np.zeros(len(vocabulary)), '1': np.zeros(len(vocabulary))}
        self.class_counts = {'0': 0, '1': 0}

    def tokenize(self, text):
        return word_tokenize(text.lower())

    def train(self, train_data):
        for _, row in train_data.iterrows():
            tokens = self.tokenize(row['text'])
            label = str(row['generated'])  # Assuming 'generated' column has values 0 or 1
            for token in tokens:
                if token in self.vocabulary:
                    self.class_word_counts[label][self.reverse_index[token]] += 1
            self.class_counts[label] += 1

    def calculate_probability(self, word_idx, label):
        return self.class_word_counts[label][word_idx] / self.class_counts[label]

    def predict(self, document):
        tokens = self.tokenize(document)
        probabilities = {label: 1.0 for label in self.class_counts.keys()}

        for label in probabilities:
            for token in tokens:
                if token in self.vocabulary:
                    word_idx = self.reverse_index[token]
                    probabilities[label] *= self.calculate_probability(word_idx, label)

        return max(probabilities, key=probabilities.get)

    def evaluate_accuracy(self, dev_data):
        correct_predictions = 0
        total_predictions = len(dev_data)

        for _, row in dev_data.iterrows():
            predicted_label = self.predict(row['text'])
            true_label = str(row['generated'])
            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def calculate_smoothed_probability(self, word_idx, label, alpha):
        numerator = self.class_word_counts[label][word_idx] + alpha
        denominator = self.class_counts[label] + alpha * len(self.vocabulary)
        return numerator / denominator

    def predict_with_smoothing(self, document, alpha):
        tokens = self.tokenize(document)
        probabilities = {label: 1.0 for label in self.class_counts.keys()}

        for label in probabilities:
            for token in tokens:
                if token in self.vocabulary:
                    word_idx = self.reverse_index[token]
                    probabilities[label] *= self.calculate_smoothed_probability(word_idx, label, alpha)

        return max(probabilities, key=probabilities.get)



# Step 4: Train the Naive Bayes Classifier
nb_classifier = NaiveBayesClassifier(vocabulary, reverse_index)
nb_classifier.train(train_data)

# Step 5: Calculate and Print Accuracy on the Development Set
dev_accuracy = nb_classifier.evaluate_accuracy(dev_data)
print(f"Accuracy on Development Set: {dev_accuracy}")


# Experiment with different smoothing values
alpha_values = [0.1, 0.5, 1.0, 2.0]

for alpha in alpha_values:
    nb_classifier_with_smoothing = NaiveBayesClassifier(vocabulary, reverse_index)
    nb_classifier_with_smoothing.train(train_data)
    # nb_classifier_with_smoothing.calculate_smoothed_probability()
    dev_accuracy_with_smoothing = nb_classifier_with_smoothing.evaluate_accuracy(dev_data)
    print(f"Accuracy on Development Set with Smoothing (alpha={alpha}): {dev_accuracy_with_smoothing}")

def get_top_words(class_word_counts, reverse_index, class_label, top_n=10):
    word_indices = np.argsort(class_word_counts[class_label])[::-1][:top_n]
    top_words = [word for word, idx in reverse_index.items() if idx in word_indices]
    return top_words

# Get top words for each class
top_words_class_0 = get_top_words(nb_classifier.class_word_counts, reverse_index, '0')
top_words_class_1 = get_top_words(nb_classifier.class_word_counts, reverse_index, '1')

print("Top Words for Class 0:", top_words_class_0)
print("Top Words for Class 1:", top_words_class_1)

# Load the test dataset (replace 'test_data.csv' with your actual test dataset file)
test_data = pd.read_csv('test_essays.csv')

# Predict labels for the test dataset using the optimal alpha value
optimal_alpha = 1.0  # Replace with the alpha value that performed the best in your experiments

# Initialize and train the classifier with the optimal alpha
nb_classifier_optimal = NaiveBayesClassifier(vocabulary, reverse_index)
nb_classifier_optimal.train(train_data)

# Make predictions on the test dataset
test_predictions = test_data['text'].apply(lambda x: nb_classifier_optimal.predict_with_smoothing(x, alpha=optimal_alpha))

# Create a Kaggle submission file
submission_df = pd.DataFrame({'id': test_data['id'], 'generated': test_predictions})
submission_df.to_csv('submission.csv', index=False)
