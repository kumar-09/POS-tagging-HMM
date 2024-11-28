import tkinter as tk
from tkinter import scrolledtext, font as tkfont
import nltk
from nltk.corpus import brown
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("The spaCy model 'en_core_web_sm' is not installed. Please run 'python -m spacy download en_core_web_sm' to install it.")
    exit()

# HMM-based POS Tagger class
class HMMPOSTagger2:
    def __init__(self):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(float)
        self.vocab = set()
        self.tags = set()

    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            prev_tag = '<START>'
            for word, tag in sentence:
                self.tag_counts[tag] += 1
                self.transition_probs[prev_tag][tag] += 1
                self.emission_probs[tag][word] += 1
                self.vocab.add(word)
                self.tags.add(tag)
                prev_tag = tag

        # Normalize probabilities
        for prev_tag in self.transition_probs:
            total = sum(self.transition_probs[prev_tag].values())
            for tag in self.transition_probs[prev_tag]:
                self.transition_probs[prev_tag][tag] /= total

        for tag in self.emission_probs:
            total = sum(self.emission_probs[tag].values())
            for word in self.emission_probs[tag]:
                self.emission_probs[tag][word] /= total

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for tag in self.tags:
            V[0][tag] = self.transition_probs['<START>'][tag] * self.emission_probs[tag].get(sentence[0], 1e-10)
            path[tag] = [tag]

        # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
            V.append({})
            newpath = {}

            for tag in self.tags:
                (prob, state) = max((V[t-1][prev_tag] * self.transition_probs[prev_tag].get(tag, 1e-10) * 
                                     self.emission_probs[tag].get(sentence[t], 1e-10), prev_tag) 
                                    for prev_tag in self.tags)
                V[t][tag] = prob
                newpath[tag] = path[state] + [tag]

            path = newpath

        (prob, state) = max((V[len(sentence) - 1][tag], tag) for tag in self.tags)
        return path[state]

    def tag(self, sentence):
        return list(zip(sentence, self.viterbi(sentence)))

    def print_transition_for_sentence(self, sentence):
        # Get the tagged sequence for the given sentence
        tagged_sequence = self.viterbi(sentence)
        transitions = defaultdict(float)

        # Compute transitions from the tagged sequence
        prev_tag = '<START>'
        for tag in tagged_sequence:
            transitions[(prev_tag, tag)] += 1
            prev_tag = tag
        
        # Normalize the transitions
        total_transitions = sum(transitions.values())
        transitions = {pair: count / total_transitions for pair, count in transitions.items()}

        print("Transition Probabilities for Given Sentence:")
        for (prev_tag, tag), prob in transitions.items():
            print(f"  From '{prev_tag}' to '{tag}': {prob:.4f}")

# Named Entity Recognition function
def preprocess_sentence_with_ner(sentence):
    doc = nlp(sentence)
    ner_tags = {ent.text: 'NOUN' for ent in doc.ents}
    return ner_tags

# Data preparation function
def prepare_data():
    tagged_sents = brown.tagged_sents(tagset='universal')
    return [[(word.lower(), tag) for word, tag in sent] for sent in tagged_sents]

# Cross-validation data splitting
def get_train_test_data(tagged_sents, fold, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    splits = list(kf.split(tagged_sents))
    train_index, test_index = splits[fold]
    train_data = [tagged_sents[i] for i in train_index]
    test_data = [tagged_sents[i] for i in test_index]
    return train_data, test_data

# Model evaluation
def evaluate_model(tagged_sents, num_folds=5):
    accuracies = []
    y_true = []
    y_pred = []
    
    all_tags = set()  # To collect all tags

    for fold in range(num_folds):
        train_data, test_data = get_train_test_data(tagged_sents, fold, num_folds)
        
        tagger = HMMPOSTagger2()
        tagger.train(train_data)
        
        true_tags = []
        predicted_tags = []
        
        for sentence in test_data:
            words, tags = zip(*sentence)
            predicted_tags_for_sentence = [tag for word, tag in tagger.tag(words)]
            
            true_tags.extend(tags)
            predicted_tags.extend(predicted_tags_for_sentence)
            all_tags.update(tags)
            all_tags.update(predicted_tags_for_sentence)
        
        accuracy = accuracy_score(true_tags, predicted_tags)
        accuracies.append(accuracy)
        
        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)
    
    avg_accuracy = np.mean(accuracies)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(all_tags))

    # Calculate per-tag precision, recall, and F1
    precision_per_tag, recall_per_tag, f1_per_tag, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(all_tags))

    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    _, _, f05, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', beta=0.5)
    _, _, f2, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', beta=2.0)

    return avg_accuracy, conf_matrix, precision, recall, f1_macro, f05, f2, list(all_tags), precision_per_tag, recall_per_tag, f1_per_tag

# Confusion matrix plotting
def plot_confusion_matrix(conf_matrix, tags):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=tags, yticklabels=tags)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot as an image file
    plt.show()

# Demo GUI Application
def train_tagger():
    data = prepare_data()
    tagger = HMMPOSTagger2()
    tagger.train(data)
    return tagger

def tag_sentence(tagger, sentence):
    words = sentence.lower().split()
    return tagger.tag(words)

def process_input(): #for GUI 
    sentence = input_text.get("1.0", tk.END).strip()
    if sentence.lower() == 'quit':
        root.quit()
    else:
        tagged_sentence = tag_sentence(tagger, sentence)
        result_text.config(state=tk.NORMAL)
        result_text.delete("1.0", tk.END)  # Clear previous output
        for word, tag in tagged_sentence:
            result_text.insert(tk.END, f"{word} ({tag})\n")
        
        # Print transition probabilities for the given sentence
        tagger.print_transition_for_sentence([tag for word, tag in tagged_sentence])
        
        result_text.config(state=tk.DISABLED)

# Main program and evaluation
def main():
    tagged_sents = prepare_data()
    avg_accuracy, conf_matrix, precision, recall, f1_macro, f05, f2, tags, precision_per_tag, recall_per_tag, f1_per_tag = evaluate_model(tagged_sents)

    # Print overall metrics to the terminal
    print(f"Average Accuracy (5-Fold Cross-Validation): {avg_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_macro:.4f}")
    print(f"F0.5-Score: {f05:.4f}")
    print(f"F2-Score: {f2:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Print per-tag metrics to the terminal
    print("\nPer-POS Tag Performance:")
    for idx, tag in enumerate(tags):
        print(f"{tag}: Precision: {precision_per_tag[idx]:.4f}, Recall: {recall_per_tag[idx]:.4f}, F1-Score: {f1_per_tag[idx]:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, tags)

if __name__ == "__main__":
    # Evaluate and show results in console
    main()

    # Create the GUI demo
    root = tk.Tk()
    root.title("HMM POS Tagger")

    # Set a background color
    root.configure(bg='#F0F8FF')

    # Custom fonts
    title_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
    label_font = tkfont.Font(family="Helvetica", size=14)
    text_font = tkfont.Font(family="Courier New", size=14)

    # Create and place widgets
    title_label = tk.Label(root, text="HMM POS Tagger", font=title_font, bg='#F0F8FF')
    title_label.pack(pady=20)

    input_label = tk.Label(root, text="Enter a sentence:", font=label_font, bg='#F0F8FF')
    input_label.pack(pady=10)

    input_text = scrolledtext.ScrolledText(root, height=8, width=70, font=text_font, wrap=tk.WORD)
    input_text.pack(pady=10)

    process_button = tk.Button(root, text="Tag Sentence", font=label_font, command=process_input, bg='#4CAF50', fg='white', relief=tk.RAISED)
    process_button.pack(pady=15)

    result_label = tk.Label(root, text="Tagged Sentence:", font=label_font, bg='#F0F8FF')
    result_label.pack(pady=10)

    result_text = scrolledtext.ScrolledText(root, height=12, width=70, font=text_font, state=tk.DISABLED, wrap=tk.WORD)
    result_text.pack(pady=10)

    # Train the tagger for the demo
    print("Training the HMM POS tagger...")
    tagger = train_tagger()
    print("Tagger trained successfully!")

    # Start the GUI event loop
    root.mainloop()