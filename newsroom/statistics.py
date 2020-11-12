#coding=utf8

INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'

def novel_ngrams(document, gold_summ, ngram):
    doc_grams = set()
    for i in range(len(document)-ngram+1):
        doc_grams.add(' '.join(document[i:i+ngram]))
    count = 0
    novel = 0
    for i in range(len(gold_summ)-ngram+1):
        gram = ' '.join(gold_summ[i:i+ngram])
        if gram not in doc_grams:
            novel += 1
        count += 1
    return novel, count

if __name__ == '__main__':
    import json
    import spacy
    nlp_seg = spacy.load("en_core_web_sm")

    source_dict = {}
    cluster_size = {}
    cluster_size_sum = 0
    cluster_num = 0

    summary_length = 0
    summary_sent_num = 0
    summary_num = 0
    document_length = 0
    document_sent_num = 0
    document_num = 0
    extract_num = 0
    from_same_year = 0

    document_vocabs = set()
    summary_vocabs = set()

    novel_unigrams_num = 0
    unigrams_num = 0
    novel_bigrams_num = 0
    bigrams_num = 0
    novel_trigrams_num = 0
    trigrams_num = 0
    novel_fgrams_num = 0
    fgrams_num = 0

    for line in open(INPUT_FILE):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)
        # Cluster num
        cluster_num += 1

        # Cluster size
        if len(pair_obj) not in cluster_size:
            cluster_size[len(pair_obj)] = 1
        else:
            cluster_size[len(pair_obj)] += 1
        cluster_size_sum += len(pair_obj)

        years = set()

        for pid in pair_obj:
            pair = pair_obj[pid]
            document = pair['[DOCUMENT]'].lower()
            summary = pair['[SUMMARY]'].lower()
            source = pair['[SORUCE]'].replace(':80', '')
            timestamp = pair['[DATE]']
            year = timestamp[:4]
            month = timestamp[4:6]
            date = timestamp[6:]
            years.add(year)

            # News source
            if source not in source_dict:
                source_dict[source] = 1
            else:
                source_dict[source] += 1

            # Document length
            document_toks = document.split()
            document_length += len(document_toks)
            document_num += 1
            doc_sentences = nlp_seg(document)
            document_sent_num += len(doc_sentences)
            document_vocabs |= set(document_toks)

            # Summary length
            summary_toks = summary.split()
            summary_length += len(summary_toks)
            summary_num += 1
            summary_sentences = nlp_seg(summary)
            summary_sent_num += len(summary_sentences)
            summary_vocabs |= set(summary_toks)

            # Novelty
            novel, count = novel_ngrams(document_toks, summary_toks, 1)
            novel_unigrams_num += novel
            unigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 2)
            novel_bigrams_num += novel
            bigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 3)
            novel_trigrams_num += novel
            trigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 4)
            novel_fgrams_num += novel
            fgrams_num += count

            # Extract summary
            if document.lower().find(' '.join(summary_toks[:min(len(summary_toks), 10)]).lower()) > -1:
                extract_num += 1

        if len(years) == 1:
            from_same_year += 1

    print ('Number for cluster:', cluster_num)
    print ('Dataset Size (number of doc-summary pairs)', cluster_size_sum)
    print ('Avg cluster size', cluster_size_sum/cluster_num)
    print ('Cluster size:', cluster_size)
    print ('Cluster from same year:', from_same_year/cluster_num)

    print ('Extract percentage:', extract_num / summary_num)
    print ('Avg Summary length (word):', summary_length / summary_num)
    print ('Avg Document length (word):', document_length / document_num)
    #print ('Avg Summary length (sentence):', summary_sent_num / summary_num)
    #print ('Avg Document length (sentence):', document_sent_num / document_num)
    print ('Document vocab size:', len(document_vocabs))
    print ('Summary vocab size:', len(summary_vocabs))
    print ('Novel unigrams:', novel_unigrams_num/unigrams_num)
    print ('Novel bigrams:', novel_bigrams_num/bigrams_num)
    print ('Novel trigrams:', novel_trigrams_num/trigrams_num)
    print ('Novel 4-grams:', novel_fgrams_num/fgrams_num)

    print ('Number of source:', len(source_dict))
    print ('\n')
    for item in sorted(source_dict.items(), key = lambda d:d[1], reverse=True):
        if item[1] < 1000:
            break
        print (item[0] + '\t' + str(item[1]))

