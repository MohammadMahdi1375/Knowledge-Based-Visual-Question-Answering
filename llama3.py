import gc
import torch
import spacy
import transformers
import string   ## For removing punctuations
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.stem import PorterStemmer
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

class llama3:
    def __init__(self)
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.use_pipeline = True
        
        if self.use_pipeline:
            self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda:0",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cuda:0")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        self.stemmer = PorterStemmer()
    

    def contextExtraction(self, system_content, user_content):
        ##### Pipeline
        if self.use_pipeline:
            input_prompt = user_content

            messages = [
                {"role": "system", "content": "Determine the main idea of this question shortly:"},
                {"role": "user", "content": input_prompt},
            ]



            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
            )

            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            final_output = outputs[0]["generated_text"][len(prompt):]
            final_output = final_output.replace('"', '')
            final_output = final_output.lower()

            pattern1 = "the main idea of this question is to "
            pattern2 = "the main idea of this question is: "
            if pattern1 in final_output:
                index = final_output.find(pattern1) + len(pattern1)
                final_output = final_output[index:len(final_output)-1]
            
                words = final_output.split(' ')
                if ('the' in words and words.index('the') >= 0 and words.index('the') <= 3):
                    final_output = final_output[final_output.find(' the ')+5:]
                elif ('a' in words and words.index('a') >= 0 and words.index('a') <= 3):
                    final_output = final_output[final_output.find(' a ')+3:]
                elif ('an' in words and words.index('an') >= 0 and words.index('an') <= 3):
                    final_output = final_output[final_output.find(' an ')+5:]
                else:
                    tempt = "FAIL"
                    final_output = final_output[final_output.find(' ')+1:]
            
        
            elif pattern2 in final_output:
                index = final_output.find(pattern2) + len(pattern2)
                final_output = final_output[index:len(final_output)-1]
            
                words = final_output.split(' ')
                if ('the' in words and words.index('the') >= 0 and words.index('the') <= 3):
                    final_output = final_output[final_output.find(' the ')+5:]
                elif ('a' in words and words.index('a') >= 0 and words.index('a') <= 3):
                    final_output = final_output[final_output.find(' a ')+3:]
                elif ('an' in words and words.index('an') >= 0 and words.index('an') <= 3):
                    final_output = final_output[final_output.find(' an ')+5:]
        

            else:
                messages = [
                    {"role": "system", "content": "extract the main words of the following question in just one phrase:"},
                    {"role": "user", "content": user_content},
                ]

                prompt = self.pipeline.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                )

                terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                answers = []
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.01,
                    top_p=0.9,
                )

                final_output = outputs[0]["generated_text"][len(prompt):].lower().replace('"', '')
        #### AutoCasaulForLM
        else:
            input_prompt = user_content
            messages = [
                {"role": "system", "content": "Determine the main idea of this question shortly:"},
                {"role": "user", "content": input_prompt},
            ]



            prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
            )

            model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda:0")
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            final_output = response.replace('"', '')
            final_output = final_output.lower()

            pattern1 = "the main idea of this question is to "
            pattern2 = "the main idea of this question is: "
            if pattern1 in final_output:
                index = final_output.find(pattern1) + len(pattern1)
                final_output = final_output[index:len(final_output)-1]
            
                words = final_output.split(' ')
                if ('the' in words and words.index('the') >= 0 and words.index('the') <= 3):
                    final_output = final_output[final_output.find(' the ')+5:]
                elif ('a' in words and words.index('a') >= 0 and words.index('a') <= 3):
                    final_output = final_output[final_output.find(' a ')+3:]
                elif ('an' in words and words.index('an') >= 0 and words.index('an') <= 3):
                    final_output = final_output[final_output.find(' an ')+5:]
                else:
                    tempt = "FAIL"
                    final_output = final_output[final_output.find(' ')+1:]
            
        
            elif pattern2 in final_output:
                index = final_output.find(pattern2) + len(pattern2)
                final_output = final_output[index:len(final_output)-1]
            
                words = final_output.split(' ')
                if ('the' in words and words.index('the') >= 0 and words.index('the') <= 3):
                    final_output = final_output[final_output.find(' the ')+5:]
                elif ('a' in words and words.index('a') >= 0 and words.index('a') <= 3):
                    final_output = final_output[final_output.find(' a ')+3:]
                elif ('an' in words and words.index('an') >= 0 and words.index('an') <= 3):
                    final_output = final_output[final_output.find(' an ')+5:]
        

            else:
                messages = [
                    {"role": "system", "content": "Determine the main idea of this question shortly:"},
                    {"role": "user", "content": input_prompt},
                ]



                prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                )

                model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda:0")
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.01,
                    top_p=0.9,
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                final_output = response.lower().replace('"', '')
        return final_output

    

    
    def answerGenerator(self, question, informations, system_content="Answer the follwoing Question in absolutely one or two words according to the following information:"):
        prompt = self.answerPromptGenerator(question, informations)
        prompt_to_be_written = prompt
        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
                ]
        ##### pipeline
        if self.use_pipeline:
            prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )

            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            answers = []
            outputs = self.pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            final_output = outputs[0]["generated_text"][len(prompt):]
            qutation_indeces = [i for i,c in enumerate(final_output) if c == "\""]
            #print(qutation_indeces)
            final_output = final_output.replace('"', '')
            
            final_output = self.remove_punctuation(final_output)
            gc.collect()
            torch.cuda.empty_cache()

            lemmatized = ' '.join([word.lemma_ for word in nlp(final_output)])
            root_word = self.stemmer.stem(final_output.lower())

            result = []
            result.append(final_output.lower())
            if lemmatized != final_output.lower():
                result.append(lemmatized)
            if root_word != final_output.lower() and root_word != lemmatized:
                result.append(root_word)
        
        ##### AutoCausalForLM
        else:
            prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
            )

            model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda:0")
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            answers = []
            

            final_output = response
            qutation_indeces = [i for i,c in enumerate(final_output) if c == "\""]
            #print(qutation_indeces)
            final_output = final_output.replace('"', '')
            
            final_output = self.remove_punctuation(final_output)
            gc.collect()
            torch.cuda.empty_cache()

            lemmatized = ' '.join([word.lemma_ for word in nlp(final_output)])
            root_word = self.stemmer.stem(final_output.lower())

            result = []
            result.append(final_output.lower())
            if lemmatized != final_output.lower():
                result.append(lemmatized)
            if root_word != final_output.lower() and root_word != lemmatized:
                result.append(root_word)
        return result




    def answerPromptGenerator(self, question, informations):
        prompt = ""
        for information in informations:
            prompt += "Information: " + information + "\n"
        prompt += "Question: " + question

        return prompt



    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)


    def inContextAnswerGenerator(self, question, informations, system_content="Infer a one or two words answer for the following question according to the following informations and format of the Question, Answer pairs:"):
        prompt = self.inContextAnswerPromptGenerator(question, informations)
        prompt_to_be_written = prompt
        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
                ]
        
        print("@"*100)
        print(prompt)

        ##### Pipeline
        if self.use_pipeline:
            prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )

            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            answers = []
            outputs = self.pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            final_output = outputs[0]["generated_text"][len(prompt):].split("Answer:")[-1].strip()
            final_output = final_output.replace('"', '')

            final_output = self.remove_punctuation(final_output)
            gc.collect()
            torch.cuda.empty_cache()

            lemmatized = ' '.join([word.lemma_ for word in nlp(final_output)])
            root_word = self.stemmer.stem(final_output.lower())

            result = []
            result.append(final_output.lower())
            if lemmatized != final_output.lower():
                result.append(lemmatized)
            if root_word != final_output.lower() and root_word != lemmatized:
                result.append(root_word)

            print(result)
            print("@"*100)
        ##### AutoCausalForLM
        else:
            prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
            )

            model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda:0")
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            answers = []

            final_output = response.split("Answer:")[-1].strip()
            final_output = final_output.replace('"', '')

            final_output = self.remove_punctuation(final_output)
            gc.collect()
            torch.cuda.empty_cache()

            lemmatized = ' '.join([word.lemma_ for word in nlp(final_output)])
            root_word = self.stemmer.stem(final_output.lower())

            result = []
            result.append(final_output.lower())
            if lemmatized != final_output.lower():
                result.append(lemmatized)
            if root_word != final_output.lower() and root_word != lemmatized:
                result.append(root_word)

            print(result)
            print("@"*100)
        return result, prompt_to_be_written




    def inContextAnswerPromptGenerator(self, question, informations):
        content = """"""
        for info in informations:
            content += "text: " + info + "\n"
        #print(content)
        messages = [
            {"role": "system", "content": "Generate 2 Question, one word noun phrase answer pairs for the following text in the format of (Question, Answer):"},
            {"role": "user", "content": content},
        ]

        if self.use_pipeline:
            prompt = self.pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
            )

            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            lines = outputs[0]["generated_text"][len(prompt):].split("\n")
            inContext_Examples = "\n".join(lines[1:])

            final_prompt = """Question: """ + question + "\n"
            for info in informations:
                final_prompt += "information: " + info + "\n"
            final_prompt += "Question, Answer pair examples:\n"
            final_prompt += inContext_Examples

        else:
            prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
            )

            model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda:0")
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            lines = response.split("\n")
            inContext_Examples = "\n".join(lines[1:])

            final_prompt = """Question: """ + question + "\n"
            for info in informations:
                final_prompt += "information: " + info + "\n"
            final_prompt += "Question, Answer pair examples:\n"
            final_prompt += inContext_Examples

        return final_prompt
