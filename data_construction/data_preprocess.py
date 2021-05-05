#coding=utf8

from processor.util import preprocess_high_freq
from processor.selection_rank import SelectRank
from processor.process_hier_one2one import HierOneToOne
from processor.process_one2one import OneToOne
from processor.process_multi2one_lead import MultiToOneLead
from processor.text_to_json import PreproTrainJson

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
    parser.add_argument("-dataset_name", default='multi')
    parser.add_argument("-input_file", default='/scratch/xxu/multi-multi/multi_multi_clean.jsonl')
    parser.add_argument("-root_dir", default='/scratch/xxu/multi-multi/raw_data/')
    parser.add_argument("-output_dir", default='/scratch/xxu/multi-multi/raw_data/')
    parser.add_argument("-tokenizer_model_path", default='')
    parser.add_argument("-potential_model_path", default='')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-mode", default='lead')
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-max_len_doc", default=500, type=int)
    parser.add_argument("-min_doc_sent_num", default=3, type=int)
    parser.add_argument("-max_len_paragraph", default=200, type=int) #for hier-transformer
    parser.add_argument('-max_len_sup', default=500, type=int)
    parser.add_argument('-max_len_summ', default=200, type=int)
    parser.add_argument("-min_summ_sent_num", default=1, type=int)
    parser.add_argument('-max_docs_in_cluster', default=5, type=int)
    parser.add_argument('-max_paragraph_in_cluster', default=50, type=int)
    parser.add_argument('-min_sentence_length', default=3, type=int)
    parser.add_argument('-max_sentence_length', default=500, type=int)
    parser.add_argument("-cls_tok", default='<s>')
    parser.add_argument("-sep_tok", default='</s>')
    parser.add_argument('-paraphrase_threshold', default=0.9, type=float)
    parser.add_argument('-coherent_threshold', default=0.6, type=float)

    parser.add_argument("-save_path", default='/scratch/xxu/multi-multi/json/')
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-test_shard_size", default=500, type=int)

    args = parser.parse_args()

    high_freq_src, high_freq_tgt = preprocess_high_freq(args.root_dir+'/train.json')
    if args.mode == 'rank':
        processor_obj = SelectRank(args, high_freq_src, high_freq_tgt)
    if args.mode == 'hier_one_to_one':
        processor_obj = HierOneToOne(args, high_freq_src, high_freq_tgt)
    if args.mode == 'one_to_one':
        processor_obj = OneToOne(args, high_freq_src, high_freq_tgt)
    if args.mode == 'multi_to_one_lead':
        processor_obj = MultiToOneLead(args, high_freq_src, high_freq_tgt)
    processor_obj.run()

    if args.mode not in ['hier_one_to_one']:
        json_obj = PreproTrainJson(args)
        json_obj.preprocess()


