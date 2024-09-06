import nltk
from nltk.corpus import brown
from hmm_pos_tagger import HMMPOSTagger  # Assuming the previous code is saved in hmm_pos_tagger.py

# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

def prepare_data():
    tagged_sents = brown.tagged_sents(tagset='universal')
    return [[(word.lower(), tag) for word, tag in sent] for sent in tagged_sents]

def train_tagger():
    data = prepare_data()
    tagger = HMMPOSTagger()
    tagger.train(data)
    return tagger

def tag_sentence(tagger, sentence):
    words = sentence.lower().split()
    return tagger.tag(words)

def main():
    print("Training the HMM POS tagger...")
    tagger = train_tagger()
    print("Tagger trained successfully!")

    while True:
        sentence = input("\nEnter a sentence to tag (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break

        tagged_sentence = tag_sentence(tagger, sentence)
        print("\nTagged sentence:")
        for word, tag in tagged_sentence:
            print(f"{word}: {tag}")

if __name__ == "__main__":
    main()