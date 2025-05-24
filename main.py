from dataPrepration import dataPrepration
from groundingDino import groundingDino
from keyWordExtraction import keyWordExtraction
from captionGenerator import VLM
from writeCaptions2Text import writeCaptions2Text
from evaluation import evaluation
from llama3 import llama3
from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import gc
import torch
import spacy
from word2number import w2n
nlp = spacy.load("en_core_web_sm")


if __name__ == '__main__':
    starting = time.time()

    dataloader = dataPrepration(dataset_name='Ok-VQA')
    imgs, questions, candidate_answers = dataloader.data_list()
    #imgs, questions = dataloader.data_list()
    print(len(questions))
    

    kw_Extractor = keyWordExtraction()
    GD = groundingDino()
    vlm = VLM()
    LLM = llama3()
    metric = evaluation()


    output_text = writeCaptions2Text("./Answers.text")
    caption_text = writeCaptions2Text("./Captions.text")

    generated_Answers = []

    total_data = 0 
    n_true = 0
    n_true2 = 0

    for i, question in tqdm(enumerate(questions[:100])):
        prompt = kw_Extractor.keyWords(question)
        img = GD.bboxPredictor(imgs[i], prompt)
        Image.fromarray(img).save("cropped_image.png")
        
        captions_Aux_Information = LLM.contextExtraction("extract the main words of the following question in just one phrase:", question)
        image = Image.open("cropped_image.png").convert("RGB")
        captions = vlm.captionGenerator("cropped_image.png", question, captions_Aux_Information)
        
        c = captions[:3]
        # answer = LLM.answerGenerator(question, c, "Answer the follwoing Question in only one word according to the provided Information:")
        
        answer, Q_A_Cap_QA = LLM.inContextAnswerGenerator(question, c)

        output_text.write(Q_A_Cap_QA, captions, question, answer, candidate_answers[i], captions_Aux_Information)
        generated_Answers.append(answer)

        flag = False
        for y_pre in answer:
            if y_pre in candidate_answers[i]:
                flag = True
                break
            elif y_pre.replace(" ", "") in candidate_answers[i]:
                flag = True
                break
            try:
                if (str(w2n.word_to_num(y_pre)) in candidate_answers[i]):
                    flag = True
                    break
            except ValueError:
                pass

        if (flag):
            n_true += 1



        flag = False
        for y_pre in answer:
            if y_pre in candidate_answers[i]:
                flag = True
                break
            elif y_pre.replace(" ", "") in candidate_answers[i]:
                flag = True
                break
            
            else:
                for tr_output in candidate_answers[i]:
                    word1 = set(y_pre.split())
                    word2 = set(tr_output.split())

                    if bool(word1.intersection(word2)):
                        flag = True
                        break
            try:
                if (str(w2n.word_to_num(y_pre)) in candidate_answers[i]):
                    flag = True
                    break
            except ValueError:
                pass

        if (flag):
            n_true2 += 1

        total_data += 1        

        output_text.writeResults(total_data, n_true, n_true2)

    Acc = metric.accuracy(generated_Answers, candidate_answers)
    print("*"* 100)
    print(Acc) 

    print("="*100)
    ending = time.time()
    print(f"Elapsed Time {ending - starting}")