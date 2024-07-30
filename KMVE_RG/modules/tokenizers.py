import json
from collections import Counter
from config import config as args
import jieba
jieba.load_userdict(args.technical_word)

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        # self.dataset_name = args.dataset_name  #
        # self.clean_report = self.clean_report
        # self.ann = json.loads(open(self.ann_path, 'r').read())
        self.ann = json.loads(open(self.ann_path, 'r', encoding="utf_8_sig").read())
        self.dict_pth = args.dict_pth
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        if self.dict_pth != ' ':
            word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
            return word_dict[0], word_dict[1]
        else:
            total_tokens = []
            split_list = ['train', 'test', 'val']
            for split in split_list:
                for example in self.ann[split]:
                    tokens = list(jieba.lcut(example['finding']))
                    for token in tokens:
                        total_tokens.append(token)
            counter = Counter(total_tokens)
            vocab = [k for k, v in counter.items()] + ['<unk>']
            token2idx, idx2token = {}, {}
            for idx, token in enumerate(vocab):
                token2idx[token] = idx + 1
                idx2token[idx + 1] = token
            return token2idx, idx2token


    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)


    def __call__(self, report):
        tokens = list(jieba.cut(report))
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_list(self, ids):
        txt = []
        for i, idx in enumerate(ids):
            if idx > 0:
                txt.append(self.idx2token[idx])
            else:txt.append('<start/end>')

        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def decode_batch_list(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode_list(ids))
        return out



