#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import json
import torch

from transformers import BartTokenizer
from tensorboardX import SummaryWriter
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

        tensorboard_log_dir = args.model_path
        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")


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
                self.tensorboard_writer.add_text('pred_str', pred_str)
                self.tensorboard_writer.add_text('gold_str', gold_str)
                self.tensorboard_writer.add_text('src_str', src_str)

        '''
        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
        '''


    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

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
