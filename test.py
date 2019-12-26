from util.json_loader import JSONFileDataLoader
import numpy as np

train_data = JSONFileDataLoader('data/train.json','/Users/yangyang/Desktop/sgns.weibo.char',prefix='train')
inputs,query_label = train_data.next_one_tf(5,5,5)
print(inputs)


# test_data = JSONFileDataLoader('data/test.json','/Users/yangyang/Desktop/sgns.weibo.char',prefix='test')
# print(test_data.rel2scope.keys())
