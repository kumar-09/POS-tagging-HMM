import nltk
from nltk.corpus import brown
from collections import defaultdict
import numpy as np
import gradio as gr

# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

class HMMPOSTagger:
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

def prepare_data():
    tagged_sents = brown.tagged_sents(tagset='universal')
    return [[(word.lower(), tag) for word, tag in sent] for sent in tagged_sents]

def train_tagger():
    data = prepare_data()
    tagger = HMMPOSTagger()
    tagger.train(data)
    return tagger

# Global variable to store the trained tagger
trained_tagger = None

def tag_sentence(sentence):
    global trained_tagger
    if trained_tagger is None:
        trained_tagger = train_tagger()
    
    words = sentence.lower().split()
    tagged = trained_tagger.tag(words)
    
    # Format the output
    output = ""
    for word, tag in tagged:
        output += f"{word} ({tag}) "
    
    return output.strip()

# Gradio interface
iface = gr.Interface(
    fn=tag_sentence,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence to tag..."),
    outputs=gr.Textbox(),
    title="POS Tagger using Hidden Markov Model",
    description="This app tags parts of speech in a given sentence using a Hidden Markov Model trained on the Brown corpus.",
    examples=[
        ["The quick brown fox jumps over the lazy dog."],
        ["I love to eat pizza and pasta."],
        ["She quickly ran to the store to buy some milk."]
    ]
)

if __name__ == "__main__":
    print("Training the HMM POS tagger... This may take a few minutes.")
    trained_tagger = train_tagger()
    print("Tagger trained successfully! Launching the Gradio interface...")
    iface.launch(share=True)