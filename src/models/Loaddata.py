import glob
import random
import torch
from torch.utils.data import Dataset

def batch_collate(batch=None):

    def _pad(data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    pad_id = batch[0].pad_id
    pre_src = [ex.src for ex in batch]
    pre_tgt = [ex.tgt for ex in batch]
    example_ids = [ex.example_id for ex in batch]

    src = torch.tensor(_pad(pre_src, pad_id))
    tgt = torch.tensor(_pad(pre_tgt, pad_id))

    mask_src = ~(src == pad_id)
    mask_tgt = ~(tgt == pad_id)

    if (batch[0].src_txt is not None):
        src_str = [ex.src_txt for ex in batch]
        tgt_str = [ex.tgt_txt for ex in batch]
        return src, tgt, mask_src, mask_tgt, src_str, tgt_str

    return src, tgt, mask_src, mask_tgt

class SummDataObj():
    def __init__(self, src, tgt, example_id, src_txt, tgt_txt, pad_id):
        self.src = src
        self.tgt = tgt
        self.src_txt = src_txt
        self.tgt_txt = tgt_txt
        self.example_id = example_id
        self.pad_id = pad_id
 
class SummDataset(Dataset):
    def __init__(self, args, corpus_type, shuffle):
        self.args = args
        self.examples = self.load_dataset(corpus_type, shuffle)
 
    def load_dataset(self, corpus_type, shuffle):
        assert corpus_type in ["train", "dev", "test"]
        examples = []

        # Sort the glob output by file name (by increasing indexes).
        pts = sorted(glob.glob(self.args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
        if pts:
            if (shuffle):
                random.shuffle(pts)
            for pt_file in pts:
                dataset = torch.load(pt_file)
                for ex in dataset:
                    examples.append(self.preprocess(ex, corpus_type == "test"))
        else:
            pt_file = args.data_path + '.' + corpus_type + '.pt'
            dataset = torch.load(pt_file)
            for ex in dataset:
                examples.append(self.preprocess(ex, corpus_type == "test"))
        return examples

    def preprocess(self, ex, is_test):
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        tgt = ex['tgt']
        src = ex['src']
        example_id = ex['example_id']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        end_id = [tgt[-1]]
        tgt = tgt[:-1][:self.args.max_pos - 1] + end_id

        if(is_test):
            src_txt = None
            tgt_txt = None

        return SummDataObj(src, tgt, example_id, src_txt, tgt_txt, self.args.pad_id)

    def __len__(self):
        return len(self.examples)
 
    def __getitem__(self, idx):
        return(self.examples[idx])
