#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import json
import torch

from transformers import BartTokenizer
from others.utils import rouge_results_to_str, test_rouge, tile


class Translator(object):

    def __init__(self, args, model):

        self.args = args
        self.model = model

        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        if args.large:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', do_lower_case=True)
        else:
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=True)

        self.gold_path = self.args.result_path + '.gold'
        self.can_path = self.args.result_path + '.candidate'
        self.raw_src_path = self.args.result_path + '.raw_src'

        if self.args.mode == 'test':
            self.gold_out_file = codecs.open(self.gold_path, 'w', 'utf-8')
            self.can_out_file = codecs.open(self.can_path, 'w', 'utf-8')
            self.src_out_file = codecs.open(self.raw_src_path, 'w', 'utf-8')


    def translate(self, batch):

        self.model.eval()

        with torch.no_grad():
            src = batch[0]
            tgt = batch[1].contiguous()
            mask_src = batch[2]
            src_strs = batch[4]
            translations = self.translate_batch(src, mask_src)

            preds = self.ids_to_toks(translations)
            golds = self.ids_to_toks(tgt)

            for i, pred in enumerate(preds):
                pred_str = pred.strip()
                gold_str = golds[i].strip()
                src_str = '\t'.join(src_strs[i])
                self.can_out_file.write(pred_str+'\n')
                self.gold_out_file.write(gold_str+'\n')
                self.src_out_file.write(src_str+'\n')


    def report_rouge(self):
        results_dict = test_rouge(self.args.temp_dir, self.can_path, self.gold_path)
        return rouge_results_to_str(results_dict)

    def ids_to_toks(self, ids):
        toks = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in ids]
        return toks

    def translate_batch(self, src, mask_src):
        with torch.no_grad():
            hypos = self.model.generate(src, 
                        beam_size=self.beam_size,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        early_stopping=True)
        return hypos
