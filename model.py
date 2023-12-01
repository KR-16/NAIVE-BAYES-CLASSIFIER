import re
import math

# Step 1: Load and Merge Data
# (Assuming you have separate lists for human and LLM essays in the training and development sets)

# Step 2: Build Vocabulary
def build_vocabulary(texts, min_occurrence=5):
    all_text = ' '.join(texts)
    words = re.findall(r'\b\w+\b', all_text.lower())  # Tokenize words using regex
    word_counts = Counter(words)

    # Filter out rare words
    vocabulary = [word for word, count in word_counts.items() if count >= min_occurrence]

    return vocabulary

# Step 3: Create Reverse Index
def create_reverse_index(vocabulary):
    reverse_index = {word: idx for idx, word in enumerate(vocabulary)}
    return reverse_index

# Step 4: Calculate Occurrence Probability
def calculate_occurrence_probability(word, all_documents):
    num_documents_with_word = sum(1 for doc in all_documents if word in doc)
    total_documents = len(all_documents)
    return num_documents_with_word / total_documents

# Step 5: Calculate Conditional Probability
def calculate_conditional_probability(word, class_documents, all_documents):
    num_class_documents_with_word = sum(1 for doc in class_documents if word in doc)
    num_class_documents = len(class_documents)
    return num_class_documents_with_word / num_class_documents

# Step 6: Calculate Conditional Probability with Laplace Smoothing
def calculate_conditional_probability_smoothed(word, class_documents, all_documents, vocabulary_size, smoothing_parameter=1):
    num_class_documents_with_word = sum(1 for doc in class_documents if word in doc)
    num_class_documents = len(class_documents)
    return (num_class_documents_with_word + smoothing_parameter) / (num_class_documents + smoothing_parameter * vocabulary_size)

# Step 7: Predict Class
def predict_class(document, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm):
    words = re.findall(r'\b\w+\b', document.lower())
    log_prob_human = math.log(occurrence_probs_human)
    log_prob_llm = math.log(occurrence_probs_llm)

    for word in words:
        if word in vocabulary:
            log_prob_human += math.log(conditional_probs_human[vocabulary[word]])
            log_prob_llm += math.log(conditional_probs_llm[vocabulary[word]])

    return "human" if log_prob_human > log_prob_llm else "llm"

# Step 8: Calculate Accuracy
def calculate_accuracy(dev_documents, dev_labels, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm):
    correct_predictions = 0

    for doc, label in zip(dev_documents, dev_labels):
        predicted_class = predict_class(doc, vocabulary, occurrence_probs_human, occurrence_probs_llm, conditional_probs_human, conditional_probs_llm)
        if predicted_class == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(dev_documents)
    return accuracy

# Step 9: Compare the Effect of Smoothing
# (Already provided the code in the previous response)

# Step 10: Derive Top 10 Words Predicting Each Class
def top_words_for_class(class_probs, vocabulary, top_n=10):
    sorted_words = sorted(vocabulary, key=lambda word: class_probs[word], reverse=True)
    return sorted_words[:top_n]

# Step 11: Using the Test Dataset
# Assuming you have separate lists for human and LLM essays in the test set
# Already provided the code in the previous response

# Step 12: Submit Results to Kaggle
# (Follow Kaggle's submission guidelines)

# Example Usage:
# (Assuming you have separate lists for human and LLM essays in the training and development sets)
human_train_essays = [...]  
llm_train_essays = [...]

human_dev_essays = [...]
llm_dev_essays = [...]

# Assuming you have already created the vocabulary, reverse index, and split the data
vocabulary = build_vocabulary(human_train_essays + llm_train_essays)

occurrence_probs_human = calculate_occurrence_probability("the", human_train_essays)
occurrence_probs_llm = calculate_occurrence_probability("the", llm_train_essays)

vocabulary_size = len(vocabulary)

conditional_probs_human = {word: calculate_conditional_probability(word, human_train_essays, human_train_essays) for word in vocabulary}
conditional_probs_llm = {word: calculate_conditional_probability(word, llm_train_essays, llm_train_essays) for word in vocabulary}

# Continue with the rest of the code for smoothed probabilities, top words, test dataset, and Kaggle submission
