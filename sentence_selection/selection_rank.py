#coding=utf8

from transformers import BartTokenizer, BertTokenizer, BertModel
from util import trunc_string
import torch.nn as nn
import json
import torch

class SelectRank():
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

        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.args.device)
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _read_one_line(self, line):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        main_docs = []; summs = []; left_docs = []
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
            main_doc, left_doc = trunc_string(document, 
                                    self.args.max_len_doc, 
                                    self.args.min_sentence_length,
                                    self.high_freq_src, 
                                    tokenizer=self.bart_tokenizer)
            summary, _ = trunc_string(summary, 
                                    self.args.max_len_summ,
                                    self.args.min_sentence_length,
                                    self.high_freq_tgt)
            if len(main_doc.split()) > self.args.min_length and len(summary.split()) > self.args.min_length:
                main_docs.append(main_doc)
                left_docs.append(left_doc)
                summs.append(summary)
            if len(main_docs) == self.args.max_docs_in_cluster:
                break
        return main_docs, left_docs, summs

    def _get_embeddings(self, main_docs, left_docs):
        main_doc_embs = []
        for doc in main_docs:
            sents = doc.split('\t')
            embs = self.bert_tokenizer(sents, padding=True, return_tensors="pt").to(self.args.device)
            outputs = self.bert_model(**embs)
            last_hidden_states = outputs.last_hidden_state
            last_hidden_states = last_hidden_states[:,0,:].squeeze(dim=1)
            main_doc_embs.append(last_hidden_states)
        left_doc_embs = []
        for doc in left_docs:
            sents = doc.split('\t')
            embs = self.bert_tokenizer(sents, padding=True, return_tensors="pt").to(self.args.device)
            outputs = self.bert_model(**embs)
            last_hidden_states = outputs.last_hidden_state
            last_hidden_states = last_hidden_states[:,0,:].squeeze(dim=1)
            left_doc_embs.append(last_hidden_states)
        return main_doc_embs, left_doc_embs

    def _sentence_rank(self, path, src_path, tgt_path):
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        with open(path) as f:
            line = f.read().strip()
            json_obj = json.loads(line)
            for line in json_obj:
                main_docs, left_docs, summs = self._read_one_line(line)
                if len(main_docs) < 2:
                    continue
                main_doc_embs, left_doc_embs = self._get_embeddings(main_docs, left_docs)
                for i in range(len(main_docs)):
                    src = self._rank_one_example(main_docs, left_docs, main_doc_embs, left_doc_embs, i)
                    tgt = summs[i]
                    fpout_src.write(src+'\n')
                    fpout_tgt.write(tgt+'\n')
            fpout_src.close()
            fpout_tgt.close()

    def _cosin_sim(self, y1, y2):
        dim_1 = y1.size(0)
        dim_2 = y2.size(0)
        y1 = torch.repeat_interleave(y1, dim_2, dim=0)
        y2 = torch.cat(dim_1*[y2])
        cos_scores = self.cos(y1, y2)
        return torch.reshape(cos_scores, (-1, dim_2))

    def _rank_one_example(self, main_docs, left_docs, main_doc_embs, left_doc_embs, idx):
        sentences = []; embeddings = []
        curr_main_doc = main_docs[idx]
        curr_main_emb = main_doc_embs[idx]
        sentences.extend(left_docs[idx])
        embeddings.append(left_doc_embs[idx])
        for i in range(len(main_docs)):
            if i == idx:
                continue
            sentences.extend(main_docs[i])
            sentences.extend(left_docs[i])
            embeddings.append(main_doc_embs[i])
            embeddings.append(left_doc_embs[i])
        cand_sentence_emb = torch.cat(embeddings)
        cosin_sim_scores = self._cosin_sim(curr_main_emb, cand_sentence_emb)
        print (curr_main_emb.size())
        print (cand_sentence_emb.size())
        print (cosin_sim_scores.size())
        print ('')

        #cand_sentence_emb = torch.transpose(cand_sentence_emb, 0, 1)
        #main_to_cands = torch.matmul(curr_main_emb, cand_sentence_emb)
        print (corh_scores.size())
        
        #sups.append(self.args.sep_tok+' '+sup_docs[i])
        #return self.args.cls_tok + ' ' + docs[idx] + '\t' + '\t'.join(sups), summs[idx]
        return main_docs[idx]

    def run(self):
        self._sentence_rank(self.train_path,
                                    self.train_src_path,
                                    self.train_tgt_path)
        self._sentence_rank(self.dev_path,
                                    self.dev_src_path,
                                    self.dev_tgt_path)
        self._sentence_rank(self.test_path,
                                    self.test_src_path,
                                    self.test_tgt_path)


