# import re
# from collections import Counter
# import math
# import pandas as pd

# # Load Data
# train_essay = pd.read_csv("train_essays.csv")
# train_prompt = pd.read_csv("train_prompts.csv")
# test_essay = pd.read_csv("test_essays.csv")

# # Merge Data
# train_data = pd.merge(train_essay, train_prompt, on="prompt_id")

# # Separate Human and LLM Essays in Training Set
# human_train_essays = train_data[train_data["is_llm"] == 0]["essay"].tolist()
# llm_train_essays = train_data[train_data["is_llm"] == 1]["essay"].tolist()

# # Separate Human and LLM Essays in Test Set
# human_test_essays = test_essay[test_essay["is_llm"] == 0]["essay"].tolist()
# llm_test_essays = test_essay[test_essay["is_llm"] == 1]["essay"].tolist()

# # Build Vocabulary
# def build_vocabulary(texts, min_occurrence=5):
#     all_text = ' '.join(texts)
#     words = re.findall(r'\b\w+\b', all_text.lower())
#     word_counts = Counter(words)

#     # Filter out rare words
#     vocabulary = [word for word, count in word_counts.items() if count >= min_occurrence]

#     return vocabulary

# # Create Reverse Index
# def create_reverse_index(vocabulary):
#     reverse_index = {word: idx for idx, word in enumerate(vocabulary)}
#     return reverse_index

# # Calculate Occurrence Probability
# def calculate_occurrence_probability(word, all_documents):
#     num_documents_with_word = sum(1 for doc in all_documents if word in doc)
#     total_documents = len(all_documents)
#     return num_documents_with_word / total_documents

# # Calculate Conditional Probability
# def calculate_conditional_probability(word, class_documents, all_documents):
#     num_class_documents_with_word = sum(1 for doc in class_documents if word in doc)
#     num_class_documents = len(class_documents)
#     return num_class_documents_with_word / num_class_documents

# # Calculate Conditional Probability with Laplace Smoothing
# def calculate_conditional_probability_smoothed(word, class_documents, all_documents, vocabulary_size, smoothing_parameter=1):
#     num_class_documents_with_word = sum(1 for doc in class_documents if word in doc)
#     num_class_documents = len(class_documents)
#     return (num_class_documents_with_word + smoothing_parameter) / (num_class_documents + smoothing_parameter * vocabulary_size)

# # Predict Class
# def predict_class(document, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm):
#     words = re.findall(r'\b\w+\b', document.lower())
#     log_prob_human = math.log(occurrence_probs_human)
#     log_prob_llm = math.log(occurrence_probs_llm)

#     for word in words:
#         if word in vocabulary:
#             log_prob_human += math.log(conditional_probs_human[vocabulary[word]])
#             log_prob_llm += math.log(conditional_probs_llm[vocabulary[word]])

#     return "human" if log_prob_human > log_prob_llm else "llm"

# # Calculate Accuracy
# def calculate_accuracy(dev_documents, dev_labels, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm):
#     correct_predictions = 0

#     for doc, label in zip(dev_documents, dev_labels):
#         predicted_class = predict_class(doc, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm)
#         if predicted_class == label:
#             correct_predictions += 1

#     accuracy = correct_predictions / len(dev_documents)
#     return accuracy

# # Step 9: Compare the Effect of Smoothing
# # (Already provided the code in the previous response)

# # Step 10: Derive Top 10 Words Predicting Each Class
# def top_words_for_class(class_probs, vocabulary, top_n=10):
#     sorted_words = sorted(vocabulary, key=lambda word: class_probs[word], reverse=True)
#     return sorted_words[:top_n]

# # Step 11: Using the Test Dataset
# # (Already provided the code in the previous response)

# # Step 12: Submit Results to Kaggle
# # (Follow Kaggle's submission guidelines)

# # Example Usage:
# # Assuming you have already created the vocabulary, reverse index, and split the data
# vocabulary = build_vocabulary(human_train_essays + llm_train_essays)

# occurrence_probs_human = calculate_occurrence_probability("the", human_train_essays)
# occurrence_probs_llm = calculate_occurrence_probability("the", llm_train_essays)

# vocabulary_size = len(vocabulary)

# conditional_probs_human = {word: calculate_conditional_probability(word, human_train_essays, human_train_essays) for word in vocabulary}
# conditional_probs_llm = {word: calculate_conditional_probability(word, llm_train_essays, llm_train_essays) for word in vocabulary}

# # Continue with the rest of the code for smoothed probabilities,
# # Step 13: Calculate Conditional Probability with Laplace Smoothing
# conditional_probs_human_smoothed = {word: calculate_conditional_probability_smoothed(word, human_train_essays, human_train_essays, vocabulary_size) for word in vocabulary}
# conditional_probs_llm_smoothed = {word: calculate_conditional_probability_smoothed(word, llm_train_essays, llm_train_essays, vocabulary_size) for word in vocabulary}

# # Example Usage:
# # Assuming you have separate lists for human and LLM essays in the development set
# human_dev_essays = [...]
# llm_dev_essays = [...]

# accuracy_dev_smoothed = calculate_accuracy(human_dev_essays + llm_dev_essays, ["human"] * len(human_dev_essays) + ["llm"] * len(llm_dev_essays), vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human_smoothed, conditional_probs_llm_smoothed)

# print(f"Accuracy on Development Dataset with Laplace Smoothing: {accuracy_dev_smoothed}")

# # Step 14: Derive Top 10 Words Predicting Each Class
# top_words_human_smoothed = top_words_for_class(conditional_probs_human_smoothed, vocabulary)
# top_words_llm_smoothed = top_words_for_class(conditional_probs_llm_smoothed, vocabulary)

# print("Top 10 words predicting 'human' class with Laplace Smoothing:", top_words_human_smoothed)
# print("Top 10 words predicting 'llm' class with Laplace Smoothing:", top_words_llm_smoothed)

# # Step 15: Using the Test Dataset
# accuracy_test_smoothed = calculate_accuracy(human_test_essays + llm_test_essays, ["human"] * len(human_test_essays) + ["llm"] * len(llm_test_essays), vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human_smoothed, conditional_probs_llm_smoothed)

# print(f"Accuracy on Test Dataset with Laplace Smoothing: {accuracy_test_smoothed}")

# # Step 16: Submit Results to Kaggle
# # (Follow Kaggle's submission guidelines)

import re
import math
import pandas as pd
from collections import Counter

# Load Data
train_essays = pd.read_csv("train_essays.csv")
train_prompts = pd.read_csv("train_prompts.csv")
test_essays = pd.read_csv("test_essays.csv")

# Merge Data
train_data = pd.merge(train_essays, train_prompts, on="prompt_id")

# Separate Human and LLM Essays in Training Set
human_train_essays = train_data[train_data["generated"] == 0]["text"].tolist()
llm_train_essays = train_data[train_data["generated"] == 1]["text"].tolist()

# Build Vocabulary
def build_vocabulary(texts, min_occurrence=5):
    all_text = ' '.join(texts)
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_counts = Counter(words)

    # Filter out rare words
    vocabulary = [word for word, count in word_counts.items() if count >= min_occurrence]

    return vocabulary

# Create Reverse Index
def create_reverse_index(vocabulary):
    reverse_index = {word: idx for idx, word in enumerate(vocabulary)}
    return reverse_index

# Calculate Occurrence Probability
def calculate_occurrence_probability(word, all_documents):
    num_documents_with_word = sum(1 for doc in all_documents if word in doc)
    total_documents = len(all_documents)
    return num_documents_with_word / total_documents

# Calculate Conditional Probability
def calculate_conditional_probability(word, class_documents, all_documents):
    num_class_documents_with_word = sum(1 for doc in class_documents if word in doc)
    num_class_documents = len(class_documents)
    return num_class_documents_with_word / num_class_documents

# Calculate Conditional Probability with Laplace Smoothing
def calculate_conditional_probability_smoothed(word, class_documents, all_documents, vocabulary_size, smoothing_parameter=1):
    num_class_documents_with_word = sum(1 for doc in class_documents if word in doc)
    num_class_documents = len(class_documents)
    return (num_class_documents_with_word + smoothing_parameter) / (num_class_documents + smoothing_parameter * vocabulary_size)

# Predict Class
def predict_class(document, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm):
    words = re.findall(r'\b\w+\b', document.lower())
    log_prob_human = math.log(occurrence_probs_human)
    log_prob_llm = math.log(occurrence_probs_llm)

    for word in words:
        if word in vocabulary:
            log_prob_human += math.log(conditional_probs_human[vocabulary[word]])
            log_prob_llm += math.log(conditional_probs_llm[vocabulary[word]])

    return "human" if log_prob_human > log_prob_llm else "llm"

# Calculate Accuracy
def calculate_accuracy(dev_documents, dev_labels, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm):
    correct_predictions = 0

    for doc, label in zip(dev_documents, dev_labels):
        predicted_class = predict_class(doc, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm)
        if predicted_class == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(dev_documents)
    return accuracy

# Build Vocabulary
vocabulary = build_vocabulary(human_train_essays + llm_train_essays)

# Create Reverse Index
reverse_index = create_reverse_index(vocabulary)

# Calculate Occurrence Probability
occurrence_probs_human = calculate_occurrence_probability("the", human_train_essays)
occurrence_probs_llm = calculate_occurrence_probability("the", llm_train_essays)

# Calculate Vocabulary Size
vocabulary_size = len(vocabulary)

# Calculate Conditional Probability
conditional_probs_human = {word: calculate_conditional_probability(word, human_train_essays, human_train_essays) for word in vocabulary}
conditional_probs_llm = {word: calculate_conditional_probability(word, llm_train_essays, llm_train_essays) for word in vocabulary}

# Step 11: Use the Test Dataset
# Since we don't have the test dataset, Kaggle will provide it during the competition.
# You'll use the trained model to make predictions on Kaggle's test set.

# Step 12: Kaggle Submission
# The following code is a placeholder. During the Kaggle competition, Kaggle will provide the test dataset.
# You'll use the trained model to predict the class of essays in the Kaggle test set.

# Placeholder for Kaggle's test set
kaggle_test_set = test_essays

# Make Predictions on Kaggle's Test Set
kaggle_test_probabilities = []

epsilon = 1e-10  # Small epsilon value to prevent math domain error

for text in kaggle_test_set['text']:
    words = re.findall(r'\b\w+\b', text.lower())
    log_prob_human = math.log(occurrence_probs_human)
    log_prob_llm = math.log(occurrence_probs_llm)

    for word in words:
        if word in vocabulary and word in conditional_probs_human and word in conditional_probs_llm:
            prob_human = max(conditional_probs_human[word], epsilon)
            prob_llm = max(conditional_probs_llm[word], epsilon)
            log_prob_human += math.log(prob_human)
            log_prob_llm += math.log(prob_llm)

    # Convert log probabilities to probabilities
    prob_human = math.exp(log_prob_human)
    prob_llm = math.exp(log_prob_llm)

    # Normalize probabilities
    total_prob = prob_human + prob_llm
    prob_human /= total_prob
    prob_llm /= total_prob

    kaggle_test_probabilities.append(prob_llm)  # Assuming 'llm' represents the generated class

# Create Submission DataFrame
submission_df = pd.DataFrame({'id': kaggle_test_set['id'], 'generated': kaggle_test_probabilities})

# Save Submission to CSV
submission_df.to_csv('submission.csv', index=False)


