"""
Character level tokenizer
"""
import spacy
from spacy.tokens import Doc


class CharacterTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        chars = [char for char in text]
        chars = [x for x in chars if x != ""]
        return Doc(self.vocab, words=chars)


@spacy.registry.tokenizers("character_tokenizer")
def create_character_tokenizer():
    def create_tokenizer(nlp):
        return CharacterTokenizer(nlp.vocab)

    return create_tokenizer
