import random
import jieba
import numpy as np
import umap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cluster
import sys
import json
sys.path.append('../')
from KMVE_RG.config import config as args


def _preprocess_text(documents):
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
    return cleaned_documents


def get_item(cls, ann, sentence_all):
    examples = ann[cls]
    print(cls, '_len:', len(examples))
    for i in range(len(examples)):
        sentence = examples[i]['finding']
        sentence_all.append({cls + '_' + str(i): sentence})


def get_all_data(split, ann_path):
    ann = json.loads(open(ann_path, 'r', encoding='utf-8-sig').read())
    sentence_all = []
    for cls in split:
        get_item(cls, ann, sentence_all)
    print('sentence_alllen:', len(sentence_all))
    data = []
    cut_data = []

    for item in sentence_all:
        sentence = list(item.values())[0]
        data.append(sentence)
        cut_data.append(' '.join(list(jieba.lcut(sentence))))
    print('data:', len(data))
    return data, sentence_all, ann, cut_data


def _check_class_nums(topics, topic_model):
    cls_num = {}
    for item in topics:
        if item not in cls_num:
            cls_num.update({item: str(item)})

    result = len(cls_num) == topic_model.get_topic_info().shape[0]
    assert result is True, 'cls_nums need to equal to topic_model.get_topic_info().shape'


def shuffle_result(topics, topic_model, ann, data, all_sentence, shuffle=False):
    _check_class_nums(topics, topic_model)
    all_data = []
    for i in range(len(data)):
        label = topics[i] + 1
        key_list = list(all_sentence[i].keys())[0].split('_')
        origin = ann[key_list[0]][int(key_list[1])]
        origin.update({'label': label})
        all_data.append(origin)
    if shuffle is True:
        random.shuffle(all_data)
        print('shuffle data complieted !')
    return all_data

if __name__ == '__main__':
    ann_path = args.ann_path
    split = ['train', 'val', 'test']
    data, all_sentence, origin_ann, cut_data, labels = get_all_data(split=split, ann_path=ann_path)
    cut_data = cut_data
    documents = pd.DataFrame({"Document": data,
                              "ID": range(len(data)),
                              "Topic": None})

    # embedding_method = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # embeddings = embedding_method.encode(data)
    count_vectorizer = CountVectorizer()
    embeddings = count_vectorizer.fit_transform(cut_data)

    # UMAP algorithm settings
    umap_model = umap.UMAP(n_neighbors=10,
                           n_components=2,
                           min_dist=0.0,
                           metric='cosine',
                           low_memory=False)

    # bandwidth = 10
    # model = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # model = hdbscan.HDBSCAN(min_cluster_size=120,
    #                                 metric='euclidean',
    #                                 cluster_selection_method='eom',
    #                                 prediction_data=True)
    # model = cluster.DBSCAN(eps=8)
    # model = cluster.AffinityPropagation(damping=0.9799)
    model = cluster.KMeans(n_clusters=5)

    umap_model.fit(embeddings, y=None)
    umap_embeddings = umap_model.transform(embeddings)
    new_embeddings = np.nan_to_num(umap_embeddings)

    # Clustering

    model.fit(umap_embeddings)
    documents['Topic'] = model.labels_
    labels = model.labels_
    probabilities = model.probabilities_
    sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
    topic_num = sizes.shape[0]
    topic_size = dict(zip(sizes.Topic, sizes.Document))

