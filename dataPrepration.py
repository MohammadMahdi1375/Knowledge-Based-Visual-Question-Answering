import os
import re
import glob
import json
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")


class  dataPrepration:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        if self.dataset_name == "my_data":
            self.img_dir = "./data/" + self.dataset_name + "/Images/"
            self.question_dir = "./data/" + self.dataset_name + "/Questions.txt"

        elif self.dataset_name == "Ok-VQA":
            self.img_file_adr = "./data/" + self.dataset_name + "/Images/"
            self.question_file_adr = "./data/" + self.dataset_name +"/OpenEnded_mscoco_val2014_questions.json"
            self.annotation_file_adr = "./data/" + self.dataset_name + "/mscoco_val2014_annotations.json"

        elif self.dataset_name == "A-OKVQA":
            self.img_file_adr = "./data/" + self.dataset_name + "/Images/"
            self.jsonfile = "./data/" + self.dataset_name +"/aokvqa_v1p0_val.json"
        
        elif self.dataset_name == "VQAv2":
            self.img_file_adr = "./data/" + self.dataset_name + "/Images/"
            self.question_file_adr = "./data/" + self.dataset_name +"/v2_OpenEnded_mscoco_val2014_questions.json"
            self.annotation_file_adr = "./data/" + self.dataset_name + "/v2_mscoco_val2014_annotations.json"

    def data_list(self):
        if self.dataset_name == "my_data":
            return self.my_data()
        elif self.dataset_name == "Ok-VQA":
            return self.OKVQA()
        elif self.dataset_name == "A-OKVQA":
            return self.AOKVQA()
        elif self.dataset_name == "VQAv2":
            return self.VQAv2()




    def my_data(self):
        imgs = glob.glob(os.path.join(self.img_dir, '*'))

        with open(self.question_dir, 'r') as file:
            lines = file.readlines()
        questions = [line.strip() for line in lines]

        return sorted(imgs, key=self.getFileName), questions,




    def OKVQA(self):
        with open(self.question_file_adr, 'r') as file:
            question_json = json.load(file)

        with open(self.annotation_file_adr, 'r') as file:
            annotation_json = json.load(file)


        questions = question_json["questions"]
        annotation = annotation_json['annotations']

        questions_df = pd.DataFrame(questions)
        annotation_df = pd.DataFrame(annotation)


        Images = []
        Questions = []
        Answers = []
        for i in range(len(questions)):
            Images.append(self.img_file_adr + 'COCO_val2014_' + str(questions[i]['image_id']).zfill(12) + '.jpg')

            Questions.append(questions[i]['question'])

            GT = []
            Answs = list(annotation_df[annotation_df["image_id"] == questions[i]['image_id']]['answers'])[0]
            for answ in Answs:
                if (answ["raw_answer"] not in GT):
                    GT.append(answ["raw_answer"])
                if (answ["answer"] not in GT):
                    GT.append(answ["answer"])
            for gt in GT:
                lemmatized = ' '.join([word.lemma_ for word in nlp(gt)]).rstrip()
                if (lemmatized not in GT):
                    GT.append(lemmatized)
                if (gt.replace(" ", "") not in gt):
                    GT.append(gt.replace(" ", ""))
            Answers.append(GT)

        dataset = pd.DataFrame({"image_adr": Images, "question": Questions, "answer": Answers})
        return Images, Questions, Answers

    

    def AOKVQA(self, test=False):
        if test:
            with open(self.jsonfile, 'r') as file:
                test = json.load(file)

            Images = []
            Questions = []
            Answers = []
            for data in test:
                Images.append(self.img_file_adr + str(data['image_id']).zfill(12) + '.jpg')
    
                Questions.append(data['question'])
    
                GT = []
                Answs = data['choices']
                for gt in Answs:
                    lemmatized = ' '.join([word.lemma_ for word in nlp(gt)]).rstrip()
                    if (lemmatized not in Answs):
                        Answs.append(lemmatized)
                Answers.append(Answs)
    
            dataset = pd.DataFrame({"image_adr": Images, "question": Questions, "answer": Answers})
        
        else:
            with open(self.jsonfile, 'r') as file:
                test = json.load(file)

            print(len(test))
            Images = []
            Questions = []
            Answers = []
            for data in test:
                Images.append(self.img_file_adr + str(data['image_id']).zfill(12) + '.jpg')

                Questions.append(data['question'])

                GT = []
                Answs = list(map(str, list(np.unique(data['direct_answers']))))
                for gt in Answs:
                    lemmatized = ' '.join([word.lemma_ for word in nlp(gt)]).rstrip()
                    if (lemmatized not in Answs):
                        Answs.append(lemmatized)
                    if (gt.replace(" ", "") not in Answs):
                        Answs.append(gt.replace(" ", ""))
                Answers.append(Answs)

            dataset = pd.DataFrame({"image_adr": Images, "question": Questions, "answer": Answers})

        print(Images[:2])
        print(Questions[:2])
        print(Answers[:2])
        return Images, Questions, Answers 
    


    def VQAv2(self):
        with open(self.annotation_file_adr, 'r') as file:
            json_annotation = json.load(file)

        with open(self.question_file_adr, 'r') as file:
            json_question = json.load(file)

        print(len(json_annotation), len(json_annotation['annotations']))
        print(len(json_question), len(json_question['questions']))

        print(json_annotation['annotations'][1])


        q = []
        question_id = []
        img_id = []
        for question in tqdm(json_question['questions']):
            q.append(question['question'])
            question_id.append(question['question_id'])
            img_id.append(question['image_id'])

        df_question = pd.DataFrame({'image_id': img_id, 'question_id': question_id, 'question': q})

        question_id = []
        img_id = []
        answers = []
        for annotation in tqdm(json_annotation['annotations']):
            question_id.append(annotation['question_id'])
            img_id.append(annotation['image_id'])
            A = []
            A.append(annotation['multiple_choice_answer'])
            for answer in annotation['answers']:
                if (answer['answer'] not in A):
                    A.append(answer['answer'])
            answers.append(A)
        df_annotation = pd.DataFrame({'image_id': img_id, 'question_id': question_id, 'answers': answers})


        Images = []
        Questions = []
        Answers = []
        i = 0
        for q_id in tqdm(question_id[:27000]):
            Images.append(self.img_file_adr + "COCO_val2014_" + str(img_id[i]).zfill(12) + '.jpg')
            Questions.append(list(df_question[df_question['question_id'] == q_id]['question'])[0])

            for gt in answers[i]:
                lemmatized = ' '.join([word.lemma_ for word in nlp(gt)]).rstrip()
                if (lemmatized not in answers[i]):
                    answers[i].append(lemmatized)
                if (gt.replace(" ", "") not in answers[i]):
                    answers[i].append(gt.replace(" ", ""))
            Answers.append(answers[i])

            i += 1

        dataset = pd.DataFrame({"image_adr": Images, "question": Questions, "answer": Answers})

        return Images, Questions, Answers
            

    def getFileName(self, path):
        return os.path.basename(path)
        
