import copy
import json
import torch
import torch.nn as nn
from transformers import BartModel, BartConfig, BartForConditionalGeneration

class FineTuneModel(nn.Module):
    def __init__(self, args, device):
        super(FineTuneModel, self).__init__()
        self.args = args
        self.finetune = args.finetune_bart

        temp_dir = args.temp_dir
        config = BartConfig(max_position_embeddings=args.max_pos,
                            dropout=args.dropout,
                            attention_dropout=args.attention_dropout)
        if(args.large):
            self.model = BartForConditionalGeneration(config).from_pretrained('facebook/bart-large', cache_dir=temp_dir)
        else:
            self.model = BartForConditionalGeneration(config).from_pretrained('facebook/bart-base', cache_dir=temp_dir)

        # ===== initialize parameters =====
        self.to(device)


    def forward(self, src, tgt, mask_src, mask_tgt, labels):
        if self.finetune:
            res = self.model(input_ids=src,
                    attention_mask=mask_src,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=mask_tgt,
                    labels=labels)
            normalization = labels[:, 1:].ne(-100).sum().item()
            return res.loss.div(float(normalization)), res.logits
        else:
            self.eval()
            with torch.no_grad():
                res = self.model(input_ids=src,
                    attention_mask=mask_src,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=mask_tgt)
            return res.logits

    def generate(self, src, beam_size, max_length, min_length, early_stopping=True):
        hypos = self.model.generate(src,
                    num_beams=beam_size,
                    max_length=max_length,
                    min_length=min_length,
                    early_stopping=early_stopping)
        return hypos
