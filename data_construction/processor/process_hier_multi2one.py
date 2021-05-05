#coding=utf8

from processor.util import trunc_string, split_paragraph
import torch.nn as nn
import json
import torch

class HierOneToOne():
    def __init__(self, args, high_freq_src, high_freq_tgt):
        self.args = args

        root_dir = self.args.root_dir
        self.train_path = root_dir + '/train.json'
        self.train_src_path = root_dir+'/multi_train_src.jsonl'
        self.train_tgt_path = root_dir+'/multi_train_tgt.jsonl'

        self.dev_path = root_dir + '/dev.json'
        self.dev_src_path = root_dir+'/multi_dev_src.jsonl'
        self.dev_tgt_path = root_dir+'/multi_dev_tgt.jsonl'

        self.test_path = root_dir + '/test.json'
        self.test_src_path = root_dir+'/multi_test_src.jsonl'
        self.test_tgt_path = root_dir+'/multi_test_tgt.jsonl'

        self.high_freq_src = high_freq_src
        self.high_freq_tgt = high_freq_tgt

    def _read_one_line(self, line):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        main_docs = []; summs = []
        sup_doc_num = len(pair_obj)-1
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            main_doc = split_paragraph(document, 
                                    self.args.max_len_paragraph, 
                                    self.args.min_sentence_length,
                                    self.high_freq_src)
            summary, _ = trunc_string(summary, 
                                    self.args.max_len_summ,
                                    self.args.min_sentence_length,
                                    self.high_freq_tgt)
            if len(main_doc) <= 0:
                continue
            if sum([len(para) for para in main_doc]) < self.args.min_doc_sent_num:
                continue
            if len(summary.split('\t')) < self.args.min_summ_sent_num:
                continue
            main_docs.append(main_doc)
            summs.append(summary)
            if len(main_docs) == self.args.max_paragraph_in_cluster:
                break
        return main_docs, summs

    def _rank_one_example(self, main_docs, idx):
        merge_docs = []
        merge_docs.extend(main_docs[idx])
        for i in range(len(main_docs)):
            if i == idx:
                continue
            merge_docs.extend(main_docs[i])
        return merge_docs

    def process(self, path, src_path, tgt_path, if_single=False):
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        with open(path) as f:
            line = f.read().strip()
            json_obj = json.loads(line)
            for line in json_obj:
                main_docs, summs = self._read_one_line(line)
                if len(main_docs) < 2:
                    continue
                for i in range(len(main_docs)):
                    if if_single:
                        src = main_docs[i]
                    else:
                        src = self._rank_one_example(main_docs, i)
                    tgt = summs[i]
                    fpout_src.write(json.dumps(src)+'\n')
                    fpout_tgt.write(tgt+'\n')
            fpout_src.close()
            fpout_tgt.close()

    def run(self):
        if_single = False
        if self.args.mode == 'single':
            if_single = True

        self.process(self.train_path,
                        self.train_src_path,
                        self.train_tgt_path,
                        if_single=if_single)
        self.process(self.dev_path,
                        self.dev_src_path,
                        self.dev_tgt_path,
                        if_single=if_single)
        self.process(self.test_path,
                        self.test_src_path,
                        self.test_tgt_path,
                        if_single=if_single)


