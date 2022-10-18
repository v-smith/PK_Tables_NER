import spacy
from spacy.tokens import Doc


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)


@spacy.registry.tokenizers("whitespace_tokenizer")
def create_whitespace_tokenizer():
    def create_tokenizer(nlp):
        return WhitespaceTokenizer(nlp.vocab)

    return create_tokenizer


#nlp = spacy.blank("en")
#nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
