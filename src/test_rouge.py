#!/usr/bin/env python
import sys
import time
import os
import re
import shutil
import pyrouge_utils

def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge_utils.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict

def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100
        # ,results_dict["rouge_su*_f_score"] * 100
    )

def report_rouge(gold_path, can_path):
    print ("Calculating Rouge")
    results_dict = test_rouge('./tmp/rouge/', can_path, gold_path)
    print (rouge_results_to_str(results_dict))

if __name__ == '__main__':
    gold_path = sys.argv[1]
    can_path = sys.argv[2]
    report_rouge(gold_path, can_path)


