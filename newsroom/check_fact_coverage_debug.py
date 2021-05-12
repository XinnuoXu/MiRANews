#encoding=utf-8

import os
import sys
import time
import json
import random
sys.path.append('../CorrFA_for_Summarizaion/')
from Evaluation import Str2Srl
from Evaluation import Srl2Tree
from Evaluation import CWeighting
from Evaluation import Correlation

os.system('export CUDA_VISIBLE_DEVICES='+sys.argv[1])
SRL_ARCHIVE_PATH='../CorrFA_for_Summarizaion/Evaluation/srl-model-2018.05.25.tar.gz'
INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
CW_GOLD_ORG_PATH = './tmp_org_'+sys.argv[1]+'.txt'
CW_GOLD_OTH_PATH = './tmp_oth_'+sys.argv[1]+'.txt'
TMP_CHECKPOINT_FILE = './tmp_'+sys.argv[1]+'.checkpoint'
CHUNK_ID = int(sys.argv[1])
CW_THREAD = -1
MAX_DOCS_IN_CLUSTER = 2
MAX_LEN_SUMM = 100
MAX_LEN_DOC = 500

srl_obj = Str2Srl.Str2Srl(SRL_ARCHIVE_PATH)
tree_obj = Srl2Tree.Srl2Tree()

def read_from_tree(tree_res):
    doc_trees = []; gold_trees = []
    for res in tree_res:
        res = json.loads(res)
        doc_id = res["doc_id"]
        gold_tree = '\t'.join(res["gold_tree"]) + "\t" + doc_id
        doc_tree = '\t'.join(res["document_trees"])
        doc_trees.append(doc_tree)
        gold_trees.append(gold_tree)
    return doc_trees, gold_trees

def cut_by_length(doc, max_length):
    sents = doc.split('\t')
    new_doc = []; length = 0
    for line in sents:
        flist = line.split()
        length += len(flist)
        if length < max_length:
            new_doc.append(line)
    return '\t'.join(new_doc)

def load_clusters(sample_num):
    cluster_num = len([l for l in open(INPUT_FILE)])
    sample_ids = None
    if sample_num > -1:
        sample_ids = [random.randint(0, cluster_num) for i in range(sample_num)]

    clusters = []
    for i, line in enumerate(open(INPUT_FILE)):
        if (sample_ids is not None) and (i not in sample_ids):
            continue
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)

        docs = []; summs = []
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = '\t'.join(pair['[DOCUMENT]'])
            source = pair['[SORUCE]'].replace(':80', '')
            if source.find('www.newser.com') > -1:
                summary = '\t'.join(pair['[TITLE]']) + '\t' + '\t'.join(pair['[SUMMARY]'])
            else:
                summary = '\t'.join(pair['[SUMMARY]'])
            document.replace('\t', ' ')
            document = cut_by_length(document, MAX_LEN_DOC)
            summary = cut_by_length(summary, MAX_LEN_SUMM)
            docs.append(document.lower())
            summs.append(summary.lower())
            if len(docs) == MAX_DOCS_IN_CLUSTER:
                break
        clusters.append((docs, summs))
    return clusters

def doc_summary_cover(doc_trees, gold_trees):
    gold_set = CWeighting.DataSet(doc_trees, gold_trees, CW_GOLD_ORG_PATH, thred_num=CW_THREAD)
    gold_set.preprocess_mult()
    coverage_dict = {}
    coverage_detail = {}
    for line in open(CW_GOLD_ORG_PATH):
        json_obj = json.loads(line.strip())
        doc_id = json_obj['doc_id']
        p_gens = json_obj['p_gens']
        gold_tree = json_obj["decoded_lst"]
        tree_txt = ' '.join(gold_tree)
        fact_scores = []
        for i, tok in enumerate(gold_tree):
            if tok[:2] == '(F':
                fact_scores.append(p_gens[i])
        coverage_dict[tree_txt] = sum(fact_scores)/len(fact_scores)
        coverage_detail[tree_txt] = fact_scores
    return coverage_dict, coverage_detail

def others_summary_cover(doc_trees, gold_tree, curr_idx):
    doc_tree_combs = []
    gold_tree_combs = []
    for i, doc_tree in enumerate(doc_trees):
        if curr_idx == i:
            continue
        doc_tree_combs.append(doc_tree)
        gold_tree_combs.append(gold_tree)
    gold_set = CWeighting.DataSet(doc_tree_combs, gold_tree_combs, CW_GOLD_OTH_PATH, thred_num=CW_THREAD)
    gold_set.preprocess_mult()
    coverage_dict = {}
    for line in open(CW_GOLD_OTH_PATH):
        json_obj = json.loads(line.strip())
        doc_id = json_obj['doc_id']
        p_gens = json_obj['p_gens']
        gold_tree = json_obj["decoded_lst"]
        tree_txt = ' '.join(gold_tree)
        fact_scores = []
        for i, tok in enumerate(gold_tree):
            if tok[:2] == '(F':
                fact_scores.append(p_gens[i])
        if tree_txt not in coverage_dict:
            coverage_dict[tree_txt] = fact_scores
        else:
            for i in range(len(fact_scores)):
                if fact_scores[i] > coverage_dict[tree_txt][i]:
                    coverage_dict[tree_txt][i] = fact_scores[i]
    merge_dict = {}; merge_detail = {}
    for tree_txt in coverage_dict:
        merge_dict[tree_txt] = sum(coverage_dict[tree_txt])/len(coverage_dict[tree_txt])
        merge_detail[tree_txt] = coverage_dict[tree_txt]
    return merge_dict, merge_detail


def highlight_one_cluster(docs, summs):
    '''
    @@ Get SRL
    '''
    srl_res = srl_obj.annotation_process(docs, summs)

    '''
    @@ Get Tree
    '''
    tree_res = tree_obj.annotation_process(srl_res)
    doc_trees, gold_trees = read_from_tree(tree_res)

    '''
    @@ Get Content Weights
    '''

    for i, gold_tree in enumerate(gold_trees):
        coverage, detail = doc_summary_cover([doc_trees[i]], [gold_tree])
        coverage_others, detail_others = others_summary_cover(doc_trees, gold_tree, i)
        if find_win_over_example(coverage, detail, coverage_others, detail_others):
            return False

    return True


def find_win_over_example(coverage, detail, coverage_others, detail_others):
    for gold_tree in coverage:
        for i in range(len(detail[gold_tree])):
            if detail_others[gold_tree][i] > 0.3 and \
                    detail_others[gold_tree][i] - detail[gold_tree][i] > 0.15:
                print (detail_others[gold_tree][i], detail[gold_tree][i])
                return True
    return False

if __name__ == '__main__':
    sample_num = -1
    clusters = load_clusters(sample_num)
    num_of_examples = len(clusters)
    each_gpu = int(num_of_examples/4+1)
    clusters = clusters[each_gpu*CHUNK_ID: each_gpu*(CHUNK_ID+1)]

    for i, (docs, summs) in enumerate(clusters):
        try:
            if_continue = highlight_one_cluster(docs, summs)
        except:
            continue
        if not if_continue:
            break

