from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer



class keyWordExtraction:
    def __init__(self):
        self.kw_model = KeyBERT()
    
    def keyWords(self, Question):
        try: 
            output = self.kw_model.extract_keywords(docs=Question, vectorizer=KeyphraseCountVectorizer())

        except ValueError:
            output = self.kw_model.extract_keywords(Question, keyphrase_ngram_range=(1, 1), stop_words=None)
        
        return self.convertListOfKeywords2String(output)

    def convertListOfKeywords2String(self, keywords):
        keyBERT_keyWords = [word[0] for word in keywords if word[1] > 0.4]
        GD_InputPrompt = ""
        for i, word in enumerate(keyBERT_keyWords):
            GD_InputPrompt += word
            if i == len(keyBERT_keyWords)-1:
                break
            GD_InputPrompt += ", "
            
        return GD_InputPrompt

