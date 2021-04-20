#coding=utf8

from transformers import BartTokenizer
from util import trunc_string
import json

class SelectLead():
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

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def _multi_in_single_out(self, path, src_path, tgt_path):
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        with open(path) as f:
            line = f.read().strip()
            json_obj = json.loads(line)

            for line in json_obj:
                flist = line.strip().split('\t')
                cluster_id = flist[0]
                summ_url = flist[1]
                pairs = flist[2]
                pair_obj = json.loads(pairs)

                docs = []; summs = []; sups = []
                sup_doc_num = len(pair_obj)-1
                max_sup_len = self.args.max_len_sup / sup_doc_num
                for pid in pair_obj:
                    pair = pair_obj[pid]
                    document = '\t'.join(pair['[DOCUMENT]'])
                    source = pair['[SORUCE]'].replace(':80', '')
                    if source.find('www.newser.com') > -1:
                        summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
                    else:
                        summary = '\t'.join(pair['[SUMMARY]'])
                    document, _ = trunc_string(document, 
                                            self.args.max_len_doc, 
                                            self.args.min_sentence_length,
                                            self.high_freq_src, 
                                            tokenizer=self.tokenizer)
                    sup_document, _ = trunc_string(document, 
                                            max_sup_len, 
                                            self.args.min_sentence_length,
                                            self.high_freq_src,
                                            tokenizer=self.tokenizer)
                    summary, _ = trunc_string(summary, 
                                            self.args.max_len_summ,
                                            self.args.min_sentence_length,
                                            self.high_freq_tgt)
                    if len(document.split()) > self.args.min_length and len(summary.split()) > self.args.min_length:
                        docs.append(document)
                        summs.append(summary)
                        sups.append(sup_document)
                    if len(docs) == self.args.max_docs_in_cluster:
                        break
                if len(docs) < 2:
                    continue
                for i in range(len(docs)):
                    src, tgt = self._prepare_sup_docs(docs, summs, sups, i)
                    fpout_src.write(src+'\n')
                    fpout_tgt.write(tgt+'\n')

            fpout_src.close()
            fpout_tgt.close()

    def _prepare_sup_docs(self, docs, summs, sup_docs, idx):
        sups = []
        for i in range(len(docs)):
            if i == idx:
                continue
            sups.append(self.args.sep_tok+' '+sup_docs[i])
        return self.args.cls_tok + ' ' + docs[idx] + '\t' + '\t'.join(sups), summs[idx]

    def run(self):
        self._multi_in_single_out(self.train_path,
                                    self.train_src_path,
                                    self.train_tgt_path)
        self._multi_in_single_out(self.dev_path,
                                    self.dev_src_path,
                                    self.dev_tgt_path)
        self._multi_in_single_out(self.test_path,
                                    self.test_src_path,
                                    self.test_tgt_path)


