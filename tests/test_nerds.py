from nerds import SentenceGetter, sent2features
import pandas as pd

data = pd.DataFrame([{"Sentence #":"Sentence: 1", "Word": "Mark", "POS": 'NNP', "Tag": "B-per"},
                         {"Sentence #":"Sentence: 1", "Word": "and", "POS": 'CC', "Tag": "O"},
                         {"Sentence #":"Sentence: 1", "Word": "John", "POS": 'NNP', "Tag": "B-per"},
                         {"Sentence #":"Sentence: 1", "Word": "are", "POS": 'VBP', "Tag": "O"},
                         {"Sentence #":"Sentence: 1", "Word": "working", "POS": 'VBG', "Tag": "O"},
                         {"Sentence #":"Sentence: 1", "Word": "at", "POS": 'IN', "Tag": "O"},
                         {"Sentence #":"Sentence: 1", "Word": "Google", "POS": 'NNP', "Tag": "B-org"},
                         {"Sentence #": "Sentence: 2", "Word": "Mark", "POS": 'NNP', "Tag": "B-per"},
                         {"Sentence #": "Sentence: 2", "Word": "and", "POS": 'CC', "Tag": "O"},
                         {"Sentence #": "Sentence: 2", "Word": "John", "POS": 'NNP', "Tag": "B-per"},
                         {"Sentence #": "Sentence: 2", "Word": "are", "POS": 'VBP', "Tag": "O"},
                         {"Sentence #": "Sentence: 2", "Word": "working", "POS": 'VBG', "Tag": "O"},
                         {"Sentence #": "Sentence: 2", "Word": "at", "POS": 'IN', "Tag": "O"},
                         {"Sentence #": "Sentence: 2", "Word": "Google", "POS": 'NNP', "Tag": "B-org"}
                     ])
getter = SentenceGetter(data)
sentence = getter.sentences

def test_sentences():
    assert len(sentence) == 7
    assert sentence[0][1] == "Mark"

def sent_word2features():
    features = [ner.sent2features(s) for s in sentence]
    end = len(features[0]) - 1
    assert len(features[0][0]) == 15
    assert len(features[0][1]) == 19
    assert features[0][0]['word.istitle()'] == True
    assert features[0][0]['postag'] == "B-per"
    assert features[0][end]['EOS'] == True