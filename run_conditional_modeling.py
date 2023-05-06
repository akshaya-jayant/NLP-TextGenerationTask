import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
import os
import pickle
import random
import time
from typing import Dict, List, Optional
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import Counter, defaultdict
import itertools
import numpy as np
import re
import json
import io,sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nltk.tokenize import word_tokenize

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup,
    T5Tokenizer,
    GPT2LMHeadModel,
    default_data_collator,
    BartModel,
    BartForConditionalGeneration,
    MBartForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Config,
    MBartTokenizer,
    BartTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer, 
    GPT2Model,
    AutoModelForCausalLM
)

main_log = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        args,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        num_sentence: int=1000000000,
        generate: Optional[bool] = False,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.tokenizer=tokenizer
        #self.block_size = args.block_size - 2 if generate==False else args.block_size -1
        self.source_block_size = args.source_block_size
        self.target_block_size = args.target_block_size
        self.bos_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        self.eos_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.pad_token_id=self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


        self.keyword_token_id=self.tokenizer.convert_tokens_to_ids("<keyword>")
        self.length_token_id=self.tokenizer.convert_tokens_to_ids("<length>")

        self.examples = []
        self.data={}

        if "json" in file_path:
            with open(file_path, encoding="utf-8") as f:
                original_data=json.load(f)

        original_data=original_data[0:args.data_size]

        element_names=["source", "target"]
        for element_name in element_names:
            self.data[element_name]=[sentence.get(element_name,"") for sentence in original_data]

            self.data[element_name+"_tokenized"]= \
                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence)) \
                        for sentence in self.data[element_name]]

        element_keyword_names=["keyword"]
        for element_name in element_keyword_names:
            self.data[element_name]=[sentence.get(element_name,[]) for sentence in original_data]

            self.data[element_name+"_tokenized"]= \
                [[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(keyword)) \
                    for keyword in keyword_list] \
                        for keyword_list in self.data[element_name]]


        element_names=["length", "position", "relative_position"]
        for element_name in element_names:
            self.data[element_name]=[sentence.get(element_name,"") for sentence in original_data]


        self.examples=[0]*len(original_data)

    def input_generate(self, args, generate=False):
        
        self.examples=[]

        keyword_pos_use_prob=args.keyword_pos_use_prob
        length_use_prob=args.length_use_prob

        keyword_list=[]
        if args.use_keyword:
            keyword_list+=[   
                {
                    "name":["keyword"],
                    "token_id": self.keyword_token_id,
                    "max_num_token": 3,
                }
            ]

        for i in range(len(self.data["source"])):
            source=self.data["source"][i]
            target=self.data["target"][i]
            target_length=self.data["length"][i]
            

            source_tokenized=self.data["source_tokenized"][i]
            target_tokenized=self.data["target_tokenized"][i]

            use_length_judge= \
                (random.random()<length_use_prob or generate)


            keyword_tokens=[]

            for keyword_dict in keyword_list:

                text_keyword = self.data[keyword_dict["name"][0]][i]
                text_keyword_tokenized = self.data[keyword_dict["name"][0]+"_tokenized"][i]
                text_keyword_position=self.data["position"][i]
                text_keyword_relative_position=self.data["relative_position"][i]

                token_id=keyword_dict["token_id"]
                max_num_token=keyword_dict["max_num_token"]

                #When training, choice 0-3 keywords from candidates.
                #When inference, choice 1 keyword from candidates.
                #If you use input_all_keyword option, input all keyword candidates. 
                if args.input_all_keyword==False:

                    num_keyword = \
                        min(int(np.random.rand()*(max_num_token+1)), len(text_keyword_tokenized)) \
                            if generate==False else \
                        min(1, len(text_keyword_tokenized))

                    random_positions=random.sample(list(range(len(text_keyword_tokenized))), num_keyword)
                    text_keyword=[text_keyword[p] for p in random_positions]
                    text_keyword_tokenized=[text_keyword_tokenized[p] for p in random_positions]
                    text_keyword_position=[text_keyword_position[p] for p in random_positions]
                    text_keyword_relative_position=[text_keyword_relative_position[p] for p in random_positions]

                text_keyword_removed=[]
                text_keyword_tokenized_removed=[]
                text_keyword_position_removed=[]
                text_keyword_relative_position_removed=[]
                for k,t,p,rp in zip(text_keyword, text_keyword_tokenized, text_keyword_position, text_keyword_relative_position):
                    find_flag=False
                    for k2, t2 in zip(text_keyword_removed, text_keyword_tokenized_removed):
                        if k in k2 or k2 in k:
                            find_flag=True
                            break
                    if find_flag==False:
                        text_keyword_removed.append(k)
                        text_keyword_tokenized_removed.append(t)
                        text_keyword_position_removed.append(p)
                        text_keyword_relative_position_removed.append(rp)
                text_keyword=text_keyword_removed
                text_keyword_tokenized=text_keyword_tokenized_removed
                text_keyword_position=text_keyword_position_removed
                text_keyword_relative_position=text_keyword_relative_position_removed       

                for keyword, keyword_tokenized, keyword_position, keyword_relative_position \
                in zip(text_keyword, text_keyword_tokenized, text_keyword_position, text_keyword_relative_position):

                    use_position_judge = \
                        random.random()<keyword_pos_use_prob
                        
                    #when specify relative position
                    if args.use_keyword_pos and ((generate and args.generate_keyword_position=="target")
                        or use_position_judge):                        
                        processed_keyword_position=\
                            int(keyword_relative_position*(100/args.position_sep_num))*args.position_sep_num

                    #when specify absolute position
                    elif args.use_keyword_abspos and ((generate and args.generate_keyword_position=="target")
                        or use_position_judge):
                        processed_keyword_position=\
                            int(keyword_position/args.position_abs_sep_num)*args.position_abs_sep_num

                    else:
                        processed_keyword_position="None"


                    keyword_tokens.append((\
                        token_id,
                        keyword_tokenized,
                        self.tokenizer.convert_tokens_to_ids(f"<keyword_pos_{processed_keyword_position}>")))


            keyword_tokens= \
                list(itertools.chain.from_iterable(\
                    [[sep_token_id] \
                    +keyword_token \
                    +([keyword_position])
                        for sep_token_id, keyword_token, keyword_position in keyword_tokens]))


            if args.use_length:
                if use_length_judge==False:
                    text_length="None"
                elif generate and args.generate_text_length=="none":
                    text_length="None"
                elif generate and args.generate_text_length=="target":
                    text_length=int(target_length/args.length_sep_num)*args.length_sep_num
                else:
                    text_length=int(target_length/args.length_sep_num)*args.length_sep_num

                length_tokens= \
                    [self.tokenizer.convert_tokens_to_ids(f"<length_{text_length}>")]                  

            else:
                length_tokens=[]

            #Bart(Enc-Dec model)
            if args.model_type=="encdec":

                source_tokenized=source_tokenized[:self.source_block_size-2-len(keyword_tokens)-len(length_tokens)]
                source_pad_length=\
                    self.source_block_size-2-len(source_tokenized)-len(keyword_tokens)-len(length_tokens)
                target_tokenized=target_tokenized[:self.target_block_size-2]
                target_pad_length=self.target_block_size-2-len(target_tokenized)

                input_ids= \
                    length_tokens \
                    +keyword_tokens \
                    +[self.bos_token_id]+source_tokenized+[self.eos_token_id] \
                    +[self.pad_token_id]*source_pad_length

                head_ids=[]

                decoder_input_ids= \
                    [self.bos_token_id]+target_tokenized+[self.eos_token_id]+ \
                    [self.pad_token_id]*target_pad_length

                target_ids= \
                    [self.bos_token_id]+target_tokenized+[self.eos_token_id]+ \
                    [args.ignore_index]*target_pad_length


                attention_mask= \
                    [1]*(self.source_block_size-source_pad_length) + [0]*source_pad_length


                decoder_attention_mask= \
                    [1]*(self.target_block_size-target_pad_length) + [0]*target_pad_length


            #GPT(decoder-only model)
            elif args.model_type=="dec":
                target_tokenized=\
                    target_tokenized[:self.target_block_size-2-len(keyword_tokens)-len(length_tokens)]
                target_pad_length=\
                    self.target_block_size-2-len(target_tokenized)-len(keyword_tokens)-len(length_tokens)

                if generate==False:
                    input_ids= \
                        length_tokens \
                        +keyword_tokens \
                        +[self.bos_token_id]+target_tokenized+[self.eos_token_id] \
                        +[self.pad_token_id]*target_pad_length
                    attention_mask= \
                        [1]*(self.target_block_size-target_pad_length) + [0]*target_pad_length

                else:
                    input_ids= \
                        length_tokens \
                        +keyword_tokens \
                        +[self.bos_token_id] 
                    attention_mask= \
                        [1]*(self.target_block_size-target_pad_length) + [0]*target_pad_length


                head_ids=[]

                decoder_input_ids = []
                decoder_attention_mask= []

                target_ids= \
                    [args.ignore_index]*len(length_tokens) \
                    +[args.ignore_index]*len(keyword_tokens) \
                    +[args.ignore_index]+target_tokenized+[self.eos_token_id] \
                    +[args.ignore_index]*target_pad_length


            self.examples.append(
                {"input_ids":input_ids,
                "head_ids":head_ids,
                "decoder_input_ids":decoder_input_ids,
                "target_ids":target_ids,
                "attention_mask":attention_mask,
                "decoder_attention_mask":decoder_attention_mask}
                )


        for i in range(0):
            print("input:", self.examples[i]["input_ids"])
            print("target:", self.examples[i]["target_ids"])
            print("decoder_input_ids:", self.examples[i]["decoder_input_ids"]) 
            print("attention:", self.examples[i]["attention_mask"])   



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:  
        return {k:torch.tensor(v, dtype=torch.long) for k,v in self.examples[i].items()}

def get_arguments():
    arg_parser = argparse.ArgumentParser()    

    arg_parser.add_argument("--train_data_file", type=str, default=None, help="train data path")
    arg_parser.add_argument("--eval_data_file", type=str, default=None, help="eval data path")
    arg_parser.add_argument("--test_data_file", type=str, default=None, help="test data path")
    arg_parser.add_argument("--min_sentence_size", type=int, default=-1, help="min sentence size")
    arg_parser.add_argument("--source_block_size", type=int, default=1024, help="max sentence size")
    arg_parser.add_argument("--target_block_size", type=int, default=128, help="max sentence size")
    arg_parser.add_argument("--local_rank", type=int, default=-1, help="local rank of logger")

    arg_parser.add_argument("--output_dir", type=str, default="./model/fine_tuned_models/tmp", help="output dir")
    arg_parser.add_argument("--do_train", action="store_true", help="do training")
    arg_parser.add_argument("--do_eval", action="store_true", help="do evaluation")
    arg_parser.add_argument("--do_generate", action="store_true", help="do generation of the sample texts")
    arg_parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="train batch size")
    arg_parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="eval and batch size")
    arg_parser.add_argument("--per_device_generate_batch_size", type=int, default=8, help="generation batch size")
    arg_parser.add_argument("--total_batch_size", type=int, default=256, help="train batch size")
    arg_parser.add_argument("--per_device_pretrain_batch_size", type=int, default=8, help="eval and batch size")
    arg_parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="graddient accumulation steps. automatically decided")
    arg_parser.add_argument("--model_scratch", action="store_true", help="reset pre-trained model params")
    arg_parser.add_argument("--cache_dir", type=str, default=None, help="cache dir of the model")
    arg_parser.add_argument("--init_word_embedding", action="store_true", help="reset word embedding params")

    arg_parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    arg_parser.add_argument("--new_learning_rate", type=float, default=1e-4, help="learning rate for initial emb")
    arg_parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    arg_parser.add_argument("--adam_beta1", type=float, default=0.9, help="optimizer params")
    arg_parser.add_argument("--adam_beta2", type=float, default=0.999, help="optimizer params")
    arg_parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="optimizer params")
    arg_parser.add_argument("--max_grad_norm", type=float, default=0.1, help="clip large gradient")
    arg_parser.add_argument("--num_train_epochs", type=int, default=3, help="epochs")
    arg_parser.add_argument("--eval_freq", type=int, default=1, help="eval frequent")

    arg_parser.add_argument("--seed", type=int, default=0, help="seed")
    arg_parser.add_argument("--n_gpu", type=int, default=1, help="gpu num")
    arg_parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    arg_parser.add_argument("--fp16", action="store_true", help="use fp16")
    arg_parser.add_argument("--ignore_index", type=int, default=-100, 
    help="ignore index of the crossentropyloss")
    arg_parser.add_argument("--data_size", type=int, default=100000000000000, help="data size")

    
    arg_parser.add_argument("--model_name_or_path", type=str, default=None, help="model name or path")
    arg_parser.add_argument("--model_type", type=str, default="encdec", help="model type. encdec/dec")
    arg_parser.add_argument("--config_name", type=str, default=None, help="config name")
    arg_parser.add_argument("--tokenizer_name", type=str, default=None, help="tokenizer name. if not specified, use tokenizer depending on model")
    arg_parser.add_argument("--use_keyword", action="store_true", help="use keyword to specify sentence. please turn on")
    arg_parser.add_argument("--input_all_keyword", action="store_true", help="Use only keyword input file when in generations")
    arg_parser.add_argument("--use_length", action="store_true", help="Use length to specify sentence")
    arg_parser.add_argument("--use_keyword_weight", action="store_true", help="Use heavey weight of keyword loss")
    arg_parser.add_argument("--use_keyword_pos", action="store_true", help="use keyword position to specify sentence")
    arg_parser.add_argument("--use_keyword_abspos", action="store_true", help="use keyword absolute position to specify sentence")
    arg_parser.add_argument("--length_sep_num", type=int, default=10)
    arg_parser.add_argument("--position_sep_num", type=int, default=10)
    arg_parser.add_argument("--position_abs_sep_num", type=int, default=3)
    arg_parser.add_argument("--keyword_pos_use_prob", type=float, default=0.95)
    arg_parser.add_argument("--length_use_prob", type=float, default=0.95)
    arg_parser.add_argument("--label_smoothing", type=float, default=0.1, 
    help="label smoothing param for loss function")

    
    arg_parser.add_argument("--generation_method", type=str, default="beam")
    arg_parser.add_argument("--num_return_sequences", type=int, default=1, help="num sentence to generate per one prompt")
    arg_parser.add_argument("--filter_keyword_sentence", action="store_true", help="filter sentence that dont include keyword. in depelopment")
    arg_parser.add_argument("--temperature", type=float, default=0.4, help="temperature of genration")
    arg_parser.add_argument("--generate_text_length", type=str, default="target", help="specify sentence length when generation")
    arg_parser.add_argument("--generate_keyword_position", type=str, default="target", help="specify keyword position when generation")
    arg_parser.add_argument("--save_generation", action="store_true", help="Save generated texts")
    arg_parser.add_argument("--num_print_generated_texts", type=int, default=-1, help="how many print generated texts")
    
    arg_parser.add_argument("--num_beams", type=int, default=4, help="beam size")
    arg_parser.add_argument("--length_penalty", type=float, default=2.0, help="length penalty")
    arg_parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="no_repeat_ngram_size")
    arg_parser.add_argument("--min_length", type=int, default=55, help="min length")
    arg_parser.add_argument("--max_length", type=int, default=140, help="max length")
    arg_parser.add_argument("--dataset_type", type=str, default=None, help="cnndm/xsum/stories")

    arguments = arg_parser.parse_args()
    return arguments

def main():
    
    arguments = get_arguments()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if arguments.local_rank in [-1, 0] else logging.WARN,
    )

    main_log.warning(
        "Local Rank : %s, with Device : %s, n_gpu: %s, isDistributed: %s, Training for 16bit: %s",
        arguments.local_rank,
        arguments.device,
        arguments.n_gpu,
        bool(arguments.local_rank != -1),
        arguments.fp16,
    )

    main_log.info("eval/train params are : %s", arguments)

    set_seed(arguments.seed)

    if arguments.config_name is None:
        arguments.config_name=arguments.model_name_or_path
    auto_config = AutoConfig.from_pretrained(arguments.config_name, cache_dir=arguments.cache_dir)
    auto_config.forced_bos_token_id = None

    if arguments.tokenizer_name is None:
        arguments.tokenizer_name=arguments.model_name_or_path

    if arguments.model_type=="encdec":
        bart_tokenizer = BartTokenizer.from_pretrained(arguments.tokenizer_name)
        
        model = BartForConditionalGeneration.from_pretrained(
            arguments.model_name_or_path,
            from_tf=bool(".ckpt" in arguments.model_name_or_path),
            config=auto_config,
        )

    elif arguments.model_type=="dec":
        bart_tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer_name)
        
        model = GPT2LMHeadModel.from_pretrained(
            arguments.model_name_or_path,
            from_tf=bool(".ckpt" in arguments.model_name_or_path),
            config=auto_config,
        )

        bart_tokenizer.add_special_tokens({"pad_token":"<|padoftext|>"})
        bart_tokenizer.add_special_tokens({"bos_token":"<|startoftext|>"})
        
        auto_config.pad_token_id=bart_tokenizer.pad_token_id
        auto_config.bos_token_id=bart_tokenizer.bos_token_id


    bart_tokenizer.add_special_tokens({"additional_special_tokens":[ "<keyword>","<length>" ]})

    for i_size in range(0,700):
        bart_tokenizer.add_special_tokens({"additional_special_tokens":[f"<length_{i_size}>"]})
    
    bart_tokenizer.add_special_tokens({"additional_special_tokens":["<length_None>"]})

    if arguments.use_keyword_pos or arguments.use_keyword_abspos:
        for state in range(0,300):
            bart_tokenizer.add_special_tokens({"additional_special_tokens":[f"<keyword_pos_{state}>"]})
    
    bart_tokenizer.add_special_tokens({"additional_special_tokens":["<keyword_pos_None>"]})

    model.resize_token_embeddings(len(bart_tokenizer))

    model=model.to(arguments.device) 

    if torch.cuda.device_count()>1:
        model=torch.nn.DataParallel(model)
    
    arguments.per_device_train_batch_size*=torch.cuda.device_count()
    arguments.per_device_eval_batch_size*=torch.cuda.device_count()
    arguments.per_device_pretrain_batch_size*=torch.cuda.device_count()
    arguments.gradient_accumulation_steps=int(arguments.total_batch_size/arguments.per_device_train_batch_size)

    main_log.info(f"Batch size for Training is : {arguments.per_device_train_batch_size}, and the gradient acc steps are : {arguments.gradient_accumulation_steps}")

    training = (
        TextDataset(arguments, tokenizer=bart_tokenizer, file_path=arguments.train_data_file) 
        if arguments.do_train else None
    )

    evaluation = (
        TextDataset(arguments, tokenizer=bart_tokenizer, file_path=arguments.eval_data_file)
        if arguments.do_eval else None
    )

    ds = (
        TextDataset(arguments, tokenizer=bart_tokenizer, file_path=arguments.eval_data_file, generate=True)
        if arguments.do_generate else None
    )
    
    if arguments.do_eval:
        evaluation.input_generate(arguments) 
    
    if arguments.do_generate:
        ds.input_generate(arguments, generate=True) 

    eval_dataloader = DataLoader(evaluation, shuffle=False, collate_fn=None, batch_size=arguments.per_device_eval_batch_size) if arguments.do_eval else None
    
    generate_dataloader = DataLoader(ds, shuffle=False, collate_fn=None, batch_size=arguments.per_device_generate_batch_size) if arguments.do_generate else None

    main_log.info("train_dataset_size: {}, eval_dataset_size: {}, test_dataset_size: {}"
        .format(len(training) if arguments.do_train else None, 
                len(evaluation) if arguments.do_eval else None,
                len(ds) if arguments.do_generate else None))
    
    required_steps= int(math.ceil(len(training)/ (arguments.per_device_train_batch_size*arguments.gradient_accumulation_steps))* arguments.num_train_epochs) if arguments.do_train else 0

    steps_for_warmup=math.ceil(required_steps*0.06)
    main_log.info(f"The entire Step size is : {required_steps}, and Warmup step size is : {steps_for_warmup}")

    dacay = ["bias", "LayerNorm.weight"]

    embedded_layer = "wte" if "gpt" in arguments.model_name_or_path else "shared" if "bart" in arguments.model_name_or_path else "xxxxxxxxxxxxxx"
    
    opt_params = [
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in dacay)],
            "weight_decay": 0.0, "lr":arguments.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if embedded_layer not in n and not any(nd in n for nd in dacay)],
            "weight_decay": arguments.weight_decay, "lr":arguments.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if embedded_layer in n and not any(nd in n for nd in dacay)],
            "weight_decay": arguments.weight_decay, "lr":arguments.new_learning_rate,
        },
    ]

    adam_opt = AdamW(opt_params, lr=arguments.learning_rate)

    linear_scheduler = get_linear_schedule_with_warmup(
        optimizer=adam_opt,
        num_warmup_steps=steps_for_warmup,
        num_training_steps=required_steps,
    )

    ce_loss=torch.nn.CrossEntropyLoss(ignore_index=arguments.ignore_index, label_smoothing=arguments.label_smoothing)
    
    os.makedirs(arguments.output_dir, exist_ok=True)

    def start_traingin(num_epoch=arguments.num_train_epochs):
                
        for epoch in range(int(num_epoch)):
            main_log.info("start epoch {}".format(epoch))
            training.input_generate(arguments)
            
            train_dl = DataLoader(training, shuffle=True, collate_fn=None, batch_size=arguments.per_device_train_batch_size)
            
            model.train()
            
            for step, batch in enumerate(train_dl):
                
                input_ids=batch["input_ids"].to(arguments.device)
                decoder_input_ids=batch["decoder_input_ids"].to(arguments.device)
                target_ids=batch["target_ids"].to(arguments.device)
                attention_mask=batch["attention_mask"].to(arguments.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(arguments.device)

                if arguments.model_type=="encdec":
                    outs=model(
                        input_ids=input_ids, 
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask)[0]
                
                elif arguments.model_type=="dec":
                    outs=model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask)[0]              

                outs = outs[:, :-1, :].contiguous()
                target_ids = target_ids[:, 1:].contiguous()
                loss=ce_loss(outs.view(-1, outs.size(-1)), target_ids.view(-1))

                if torch.cuda.device_count()>1:
                    loss = loss.mean()
                
                loss = loss / arguments.gradient_accumulation_steps
                loss.backward()
                
                if (step+1) % arguments.gradient_accumulation_steps == 0 or step == len(train_dl) - 1:
                    adam_opt.step()
                    linear_scheduler.step()
                    adam_opt.zero_grad()
                    progress_bar.update(1)
            
            if (epoch+1)%arguments.eval_freq==0:
                start_evaluation()
    
    def start_evaluation():
        model.eval()
        loss_save = []
        
        for step, batch in tqdm(enumerate(eval_dataloader)):
            with torch.no_grad():
                input_ids=batch["input_ids"].to(arguments.device)
                decoder_input_ids=batch["decoder_input_ids"].to(arguments.device)
                target_ids=batch["target_ids"].to(arguments.device)
                attention_mask=batch["attention_mask"].to(arguments.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(arguments.device)

                if arguments.model_type=="encdec":
                    outs=model(
                        input_ids=input_ids, 
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask)[0]
                
                elif arguments.model_type=="dec":
                    outs=model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask)[0]

                outs = outs[..., :-1, :].contiguous()
                target_ids = target_ids[..., 1:].contiguous()
    
            loss = ce_loss(outs.view(-1, outs.size(-1)), target_ids.view(-1)).view(-1)
            loss_save.append(loss)

        loss_save = torch.cat(loss_save)
        
        try:
            per = math.exp(torch.mean(loss_save))
        except OverflowError:
            per = float("inf")
        
        final_output = {"perplexity": per}

        eval_out_fs = os.path.join(arguments.output_dir, "eval_results_lm.txt")
        
        with open(eval_out_fs, "a") as writer:
            main_log.info("Evalution Results are : ")
            
            for k in sorted(final_output.keys()):
                main_log.info("  %s = %s", k, str(final_output[k]))
                writer.write("%s = %s\n" % (k, str(final_output[k])))

    def start_generation():

        options = {
            "cnndm": {"num_beams":4, "min_length":55, "max_length":140, 
                "no_repeat_ngram_size":3, "length_penalty":2.0},
            "xsum": {"num_beams":6, "min_length":10, "max_length":60, 
                "no_repeat_ngram_size":3, "length_penalty":1.0},
            "stories": {},
        }[arguments.dataset_type]

        if "num_beams" in options and arguments.num_return_sequences>options["num_beams"]:
            options["num_beams"]=arguments.num_return_sequences

        model.eval()

        final_data=[]
        bos_token_id = ds.bos_token_id
        eos_token_id = ds.eos_token_id

        for step, batch in tqdm(enumerate(generate_dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(arguments.device)
                attention_mask=batch["attention_mask"].to(arguments.device)
                
                if arguments.model_type=="encdec":
                    outputs=model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        num_return_sequences=arguments.num_return_sequences,
                        **options
                    )

                    control_ids = []
                    
                    for text in input_ids:
                        bos_index=[i for i in range(len(text)) if text[i]==bos_token_id][0]
                        control_ids.append(text[:bos_index])
                    
                    outputs = outputs      

                elif arguments.model_type=="dec":
                    outputs=model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        top_p=0.95, 
                        max_length=128, 
                        temperature=arguments.temperature,
                        no_repeat_ngram_size=None,
                        num_return_sequences=arguments.num_return_sequences)

                    control_ids = []
                    
                    for text in outputs:
                        bos_index=[i for i in range(len(text)) if text[i]==bos_token_id][0]
                        control_ids.append(text[:bos_index])
                    
                    outputs = [text[bos_index:] for text in outputs]
            
            begin = step * arguments.per_device_generate_batch_size
            last = (step+1) *arguments.per_device_generate_batch_size 
            
            hypothesis=bart_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            ref = list(itertools.chain.from_iterable([[text]*arguments.num_return_sequences for text in ds.data["target"][begin:last]]))
            
            for i in range(len(hypothesis)):
                tokens = [control_ids[int(i/arguments.num_return_sequences)]]

                tokens = bart_tokenizer.batch_decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                    
                words_split = tokens.split("<keyword>")

                if arguments.use_length:
                    length=re.search("<length_(.+?)>", words_split[0]).group(1)
                else:
                    length="None"

                main_words=[]
                
                for tokens in words_split[1:]:
                    match = re.search("(.*)<keyword_pos_(.+?)>", tokens)
                    main_words.append({
                        "token":match.group(1).strip().split(), 
                        "position":match.group(2),
                    })
                    
                final_data.append({
                    "reference": ref[i],
                    "hypothesis": hypothesis[i],
                    "reference_length": length,
                    "reference_keyword": main_words,
                })


        with open(os.path.join(arguments.output_dir, "generated_data.json"), "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)  


    if arguments.do_train:
        progress_bar = tqdm(range(required_steps))
        main_log.info("Evalution function called")
        start_evaluation()

        if arguments.num_train_epochs>0:
            main_log.info("Training function called")
            start_traingin(num_epoch=arguments.num_train_epochs)


    if arguments.do_train==False and arguments.do_eval and arguments.do_generate==False:
        main_log.info("Evalution Running")
        start_evaluation()

    if torch.cuda.device_count()>1:
        model=model.module
        arguments.device="cuda:0"
        model=model.to(arguments.device)

    if arguments.do_generate:
        main_log.info("Genration Running")        
        start_generation()

    if arguments.do_train and arguments.output_dir is not None:
        model.save_pretrained(arguments.output_dir)
        bart_tokenizer.save_pretrained(arguments.output_dir) 


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
