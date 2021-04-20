#coding=utf8

from util import preprocess_high_freq
from selection_lead import SelectLead
from selection_rank import SelectRank
import argparse
import torch
import sys

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", default='/scratch/xxu/multi-multi/multi_multi_clean.jsonl')
    parser.add_argument("-root_dir", default='/scratch/xxu/multi-multi/raw_data/')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-mode", default='lead')
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-max_len_doc", default=500, type=int)
    parser.add_argument('-max_len_sup', default=500, type=int)
    parser.add_argument('-max_len_summ', default=200, type=int)
    parser.add_argument('-min_length', default=3, type=int)
    parser.add_argument('-max_docs_in_cluster', default=5, type=int)
    parser.add_argument('-min_sentence_length', default=3, type=int)
    parser.add_argument('-max_sentence_length', default=500, type=int)
    parser.add_argument("-cls_tok", default='<s>')
    parser.add_argument("-sep_tok", default='</s>')
    parser.add_argument('-paraphrase_threshold', default=0.9, type=float)
    parser.add_argument('-coherent_threshold', default=0.6, type=float)
    args = parser.parse_args()

    high_freq_src, high_freq_tgt = preprocess_high_freq(args.root_dir+'/train.json')
    if args.mode == 'lead':
        selector_obj = SelectLead(args, high_freq_src, high_freq_tgt)
    if args.mode == 'rank':
        selector_obj = SelectRank(args, high_freq_src, high_freq_tgt)
    selector_obj.run()

'''
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(DEVICE)
    inputs = tokenizer(["hello, my dog is cute", "hello, my dog is cute test"], padding=True, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print (last_hidden_states[:,0,:].squeeze().size())
'''
