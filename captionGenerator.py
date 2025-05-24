#######################################################################################################################
####################################################### Library #######################################################
#######################################################################################################################
import re
import string
##### 1-1) NEXT-LLaVA libraries
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

##### 1-2) Instruct-BLIP libraries
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import gc
from PIL import Image
import requests

##### 1-3) Similarity checking of theCaption Generator models
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

import nltk
from nltk.tokenize import sent_tokenize

# Download the necessary NLTK data
nltk.download('punkt')

####################################################################################################################### Added
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
####################################################################################################################### Added


class VLM:
    def __init__(self):
        self.model_path = "liuhaotian/llava-v1.6-vicuna-7b"
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        ##### 3-2) Instruct-BLIP Model Loading
        
        self.Blip_path ="/home/m_m58330/.cache/huggingface/hub/models--Salesforce--instructblip-vicuna-7b/snapshots/9e554774e5c76eb8d1b90fac042e4e6385a3e7c3"
        self.Blip_path = "/home/m_m58330/.cache/huggingface/hub/models--Salesforce--instructblip-vicuna-7b/snapshots/52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92"
        self.BLIP_model = InstructBlipForConditionalGeneration.from_pretrained(self.Blip_path)
        self.BLIP_processor = InstructBlipProcessor.from_pretrained(self.Blip_path)
        self.BLIP_model.to(self.device)
        
        ##### LLaVA
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, get_model_name_from_path(self.model_path)
        )
    

    def image_parser(self, args):
        out = args.image_file.split(args.sep)
        return out


    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image


    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out




    def LLAVA_eval_model(self, args):
        qs = args.query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN


        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, self.image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = self.image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" +qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_files = self.image_parser(args)
        images = self.load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return outputs




    def captionGenerator(self, img_adr, question, keyPhrase):
        #self.reference = keyPhrase
        self.reference = question
        image = Image.open(img_adr).convert("RGB")

       
        #prompt = "Tell me about the " + keyPhrase + " in the image in details in short sentences:"     0.47
        #prompt = "Tell me about the " + keyPhrase + ":"
        prompt_Cap = "Generate a caption about the " + self.reference + " in this image:"
        ## OK-VQA ############################################################## 0.43
        prompt = "Tell me about the " + self.reference + " in the image in details:"
        #prompt_Cap = "Generate a caption about the " + keyPhrase + " in this image:"
        #############################################################################
        # prompt = "Talk about " + keyPhrase + " in the image in details:"
        # prompt_Cap = "Generate multiple captions about the " + keyPhrase + " in this image in details in one sentence"

        
        ##################################### LLaVA #####################################
        
        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": img_adr,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 256
        })()

        llava_detailed_output = self.LLAVA_eval_model(args)

        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": prompt_Cap,
            "conv_mode": None,
            "image_file": img_adr,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 256,
        })()
        llava_caption_output = self.LLAVA_eval_model(args)
        

        ##################################### BLIP ######################################
        
        inputs = self.BLIP_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.BLIP_model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        BLIP_detailed_output = self.BLIP_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


        inputs = self.BLIP_processor(images=image, text=prompt_Cap, return_tensors="pt").to(self.device)
        outputs = self.BLIP_model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        blip_caption_output = self.BLIP_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        
        return self.sortingCaptions(llava_detailed_output, llava_caption_output, BLIP_detailed_output, blip_caption_output)

        






    def sortingCaptions(self, llava_detailed_output, llava_caption_output, BLIP_detailed_output, blip_caption_output):
        Captions = []
        
        ## OK-VQA ######################################################################################################
        
        candidates = {f'LLaVA-{i}': s.strip() for i, s in enumerate(llava_detailed_output.split(".")) if s.strip()}
        
        candidates.update({f'BLIP-{i}': s.strip() for i, s in enumerate(BLIP_detailed_output.split(".")) if s.strip()})
        
        ################################################################################################################
        #candidates = {f'LLaVA-{i}': s for i, s in enumerate(sent_tokenize(llava_detailed_output))}
        #candidates.update(({f'BLIP-{i}': s for i, s in enumerate(sent_tokenize(BLIP_detailed_output))}))


        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        candidates_embeddings = model.encode(list(candidates.values()))
        reference_embedding = model.encode(self.reference)

        max_cosine = -np.inf
        candidate_similarity_score = {}
        for i in range(candidates_embeddings.shape[0]):
            A = reference_embedding
            B = candidates_embeddings[i,:]
            cosine = np.dot(A,B)/(norm(A)*norm(B))
            candidate_similarity_score[list(candidates.values())[i]] = (cosine, list(candidates.keys())[i])
            if (cosine > max_cosine):
                max_cosine = cosine
                Auxilari_information = list(candidates.values())[i]


        sorted_candidates_score1 = {k: v for k, v in sorted(candidate_similarity_score.items(), key=lambda item: item[1][0])}
        
        ## OK-VQA #####################################################################################################
        
        candidates = {f'LLaVA-{i}': s.strip() for i, s in enumerate(llava_caption_output.split(".")) if s.strip()}
    
        candidates.update({f'BLIP-{i}': s.strip() for i, s in enumerate(blip_caption_output.split(".")) if s.strip()})


        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        candidates_embeddings = model.encode(list(candidates.values()))
        reference_embedding = model.encode(self.reference)

        max_cosine = -np.inf
        candidate_similarity_score = {}
        for i in range(candidates_embeddings.shape[0]):
            A = reference_embedding
            B = candidates_embeddings[i,:]
            cosine = np.dot(A,B)/(norm(A)*norm(B))
            candidate_similarity_score[list(candidates.values())[i]] = (cosine, list(candidates.keys())[i])
            if (cosine > max_cosine):
                max_cosine = cosine
                Auxilari_information = list(candidates.values())[i]


        sorted_candidates_score2 = {k: v for k, v in sorted(candidate_similarity_score.items(), key=lambda item: item[1][0])}

        
        for i in range(len(sorted_candidates_score2)):
            if i < 2:
                Captions.append(self.remove_leading_punctuations_and_spaces(list(sorted_candidates_score2.keys())[len(sorted_candidates_score2)-i-1]))
        for i in range(len(sorted_candidates_score1)):
            if i < 3:
                Captions.append(self.remove_leading_punctuations_and_spaces(list(sorted_candidates_score1.keys())[len(sorted_candidates_score1)-i-1]))
    

        gc.collect()
        torch.cuda.empty_cache()
        return Captions



    def remove_leading_punctuations_and_spaces(self, text):
        chars_to_remove = set(string.whitespace + string.punctuation)
        start = 0
        while start < len(text) and text[start] in chars_to_remove:
            start += 1
        return text[start:]

