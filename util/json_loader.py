import json
import os
import re
import multiprocessing
import numpy as np
import random
import gensim
import jieba

class FileDataLoader:
    def next_batch(self, B, N, K, Q):
        '''
        B: batch size.
        N: the number of relations for each batch
        K: the number of support instances for each relation
        Q: the number of query instances for each relation
        return: support_set, query_set, query_label
        '''
        raise NotImplementedError

class JSONFileDataLoader(FileDataLoader):

    def __init__(self,path,word_vec_file_name,max_length = 15,cuda = False):
        self.max_length = max_length
        file_lists = self.get_file_path(path)
        self.features = self.load_all_data(file_lists)
        self.train_features, self.test_features = self.few_shot_feature(self.features)
        self.word2id = {}
        #加载词向量
        if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
            raise Exception("[ERROR] Word vector file doesn't exist")
        self.word_vec = gensim.models.KeyedVectors.load_word2vec_format(word_vec_file_name)
        self.word2id = self.word_dic()

    def get_file_path(self,path):
        file_lists = []
        files = os.listdir(path)
        for file in files:
            if not os.path.isdir(file):
                new_path = os.path.join(path, file)
                if os.path.isdir(new_path):
                    for filename in os.listdir(new_path):
                        file_path = os.path.join(new_path, filename)
                        file_lists.append(file_path)
        return file_lists

    def load_json_data(self,filename, features):
        with open(filename, 'r') as f:
            data = json.load(f)
        for examples in data['rasa_nlu_data']['common_examples']:
            if examples['intent'] not in features:
                features[examples['intent']] = [examples['text']]
            else:
                features[examples['intent']].append(examples['text'])

    def load_all_data(self,filenames):
        features = {}
        for filename in filenames:
            self.load_json_data(filename, features)
        return features

    def few_shot_feature(self,features):
        few_shot_features = {}
        few_shot_features['机票'] = features.pop('机票')
        few_shot_features['火车'] = features.pop('火车')
        few_shot_features['酒店'] = features.pop('酒店')
        return features,few_shot_features

    def regex_sen(self,sen):
        return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+",'',sen)

    def word_dic(self):
        for i in range(self.word_vec.vectors.shape[0]):
            self.word2id[self.word_vec.wv.index2word[i]] = i
        self.word2id['UNK'] = self.word_vec.vectors.shape[0]
        self.word2id['BLANK'] = self.word_vec.vectors.shape[0] + 1

    def gen_dataset(self,features,prefix = 'train'):
        dataset, labels = [], []
        pre_shuffle = []
        for key, value in features.items():
            for item in value:
                pre_shuffle.append((jieba.lcut(self.regex_sen(item)),key))
        random.shuffle(pre_shuffle)
        for item in pre_shuffle:
            dataset.append(item[0])
            labels.append(item[1])

        self.word_vec = np.load((self.word_vec.vectors.shape[0],self.word_vec.vectors.shape[1]),dtype = np.float32)
        self.data_word = np.zeros((len(features),self.max_length),dtype = np.int32)
        self.data_length = np.zeros(len(features))

        self.rel2scope = {}#
        i = 0
        for idx,item in enumerate(dataset):
            # self.rel2scope[labels]
            #padding
            for i,value in enumerate(item):
                if i < self.max_length:
                    if i in self.word2id:
                        self.data_word[idx][i] = value
                    else:
                        self.data_word[idx][i] = self.word_vec.vectors.shape[0]
            for i in range(i+1,self.max_length):
                self.data_word[idx][i] = self.word_vec.vectors.shape[0] + 1
            self.data_length[i] = len(item)

        base_path = '../data/processed_data'
        np.save(os.path.join(base_path,prefix+'_sen.npy'),self.data_word)
        np.save(os.path.join(base_path, prefix + '_length.npy'), self.data_length)


    def _load_processed_file(self,prefix):
        base_path = '../data/processed_data'
        if not os.path.isdir(base_path):
            False
        #hahaha
        sentence_file_name = os.path.join(base_path,prefix + '_sen.py')
        length_file_name = os.path.join(base_path, prefix + '_sen.py')












