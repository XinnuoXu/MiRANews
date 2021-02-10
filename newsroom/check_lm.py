#encoding=utf-8

import os
import sys
import time
import json
import random

INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
MAX_DOCS_IN_CLUSTER = 5
MAX_LEN_SUMM = 200
MAX_LEN_DOC = 500
source_domain = 'cnn'
target_domain = 'bbc'

def cut_by_length(doc, max_length):
    sents = doc.split('\t')
    new_doc = []; length = 0
    for line in sents:
        flist = line.split()
        length += len(flist)
        if length < max_length:
            new_doc.append(line)
    return '\t'.join(new_doc)

def load_clusters():
    res_pairs = []
    res_source_single = []
    res_target_single = []
    for i, line in enumerate(open(INPUT_FILE)):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        source_domain_summary = ''
        target_domain_summary = ''
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            document = cut_by_length(document, MAX_LEN_DOC)
            summary = cut_by_length(summary, MAX_LEN_SUMM)
            if source.find(source_domain) > -1:
                source_domain_summary = summary
            if source.find(target_domain) > -1:
                target_domain_summary = summary
        if source_domain_summary != '' and target_domain_summary != '':
            res_pairs.append((source_domain_summary, target_domain_summary))
        elif source_domain_summary != '':
            res_source_single.append(source_domain_summary)
        elif target_domain_summary != '':
            res_target_single.append(target_domain_summary)
    return res_pairs, res_source_single, res_target_single

if __name__ == '__main__':
    res_pairs, res_source_single, res_target_single = load_clusters()
    random.shuffle(res_source_single)
    random.shuffle(res_target_single)
    random.shuffle(res_pairs)

    fpout = open('check_lm_train.txt', 'w')
    fpout.write('\n'.join(res_source_single))
    fpout.close()

    fpout = open('check_lm_test_same.txt', 'w')
    fpout.write('\n'.join([pair[0] for pair in res_pairs]))
    fpout.close()

    fpout = open('check_lm_test_cross.txt', 'w')
    fpout.write('\n'.join([pair[1] for pair in res_pairs]))
    fpout.close()
