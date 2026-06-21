import sys

# 手动下载exts文件夹中内容并添加path
# 手动下载ffmpeg
import numpy as np
from torch.nn import functional as F

sys.path.append(r"F:\20220909-1400-Customized-MSAFET")
sys.path.append(r"F:\20220909-1400-Customized-MSAFET\aligners")
sys.path.append(r"F:\20220909-1400-Customized-MSAFET\ASD")
sys.path.append(r"F:\20220909-1400-Customized-MSAFET\example_configs")
sys.path.append(r"F:\20220909-1400-Customized-MSAFET\extractors")
sys.path.append(r"F:\20220909-1400-Customized-MSAFET\models")
sys.path.append(r"F:\ffmpeg\bin")

print(sys.path)

import os

from main import FeatureExtractionTool
# from MSA_FET import FeatureExtractionTool


assert os.path.exists("./EATD-Corpus/t_1/positive.wav")
assert os.path.exists("./EATD-Corpus/t_1/negative.wav")
assert os.path.exists("./EATD-Corpus/t_1/neutral.wav")
# assert os.path.exists("./SIMS-Corpus/0001.mp4")

EXTRACT_TEXT = True

EXTRACT_QUESTION = True

if EXTRACT_TEXT:
    from transformers import BertTokenizer, TFBertModel

    tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-8_H-512')
    model = TFBertModel.from_pretrained("uer/chinese_roberta_L-8_H-512")

    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)

    polarities = ['negative', 'positive', 'neutral']

    max_index = 200
    eatd_path = './EATD-Corpus'


    def extract_text_feature(prefix):

        feature_lists = []
        length_list = []

        for i in range(max_index):
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                feature_per = []
                for p_idx, polarity in enumerate(polarities):
                    text_file = f'{dir_path}/{polarity}.txt'
                    text_feature = (fet.run_single(None, text_file=text_file)['text'])
                    print(f'{polarity} shape', text_feature.shape)

                    feature_per.append(text_feature)
                    length_list.append(text_feature.shape[0])
                feature_lists.append(feature_per)

            else:
                print(f'pass {prefix}_{i}')
                continue

        return feature_lists, length_list


    def pad_text_feature(feature_list, max_len):
        padded_feature_list = []

        for pnn_feature in feature_list:
            padded_pnn_list = []
            for feature in pnn_feature:
                padded_feature = np.pad(feature, ((0, (max_len - feature.shape[0])), (0, 0)), mode="constant")
                padded_pnn_list.append(padded_feature)

            padded_feature_list.append(np.array(padded_pnn_list))

        padded_feature = np.array(padded_feature_list)

        return padded_feature


    train_feature_list, train_length_list = extract_text_feature('t')
    valid_feature_list, valid_length_list = extract_text_feature('v')

    max_len = max(train_length_list + valid_length_list)

    train_features = pad_text_feature(train_feature_list, max_len)
    valid_features = pad_text_feature(valid_feature_list, max_len)

    print('train_features', train_features.shape)
    print('valid_features', valid_features.shape)

    np.save('./EATD-Features/train_text_sentbert_768.npy', train_features)
    np.save('./EATD-Features/valid_text_sentbert_768.npy', valid_features)