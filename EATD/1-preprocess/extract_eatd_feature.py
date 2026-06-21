import sys

# 手动下载exts文件夹中内容并添加path
# 手动下载ffmpeg
import numpy as np
from torch.nn import functional as F

# sys.path.append(r"F:\20220909-1400-Customized-MSAFET")
# sys.path.append(r"F:\20220909-1400-Customized-MSAFET\aligners")
# sys.path.append(r"F:\20220909-1400-Customized-MSAFET\ASD")
# sys.path.append(r"F:\20220909-1400-Customized-MSAFET\example_configs")
# sys.path.append(r"F:\20220909-1400-Customized-MSAFET\extractors")
# sys.path.append(r"F:\20220909-1400-Customized-MSAFET\models")
sys.path.append(r"D:\[SYNC]存档\[2023.03.01]DepressionDetection\20221025-Processed-EATD-feature")
sys.path.append(r"D:\[SYNC]存档\[2023.03.01]DepressionDetection\20221025-Processed-EATD-feature\aligners")
sys.path.append(r"D:\[SYNC]存档\[2023.03.01]DepressionDetection\20221025-Processed-EATD-feature\ASD")
sys.path.append(r"D:\[SYNC]存档\[2023.03.01]DepressionDetection\20221025-Processed-EATD-feature\example_configs")
sys.path.append(r"D:\[SYNC]存档\[2023.03.01]DepressionDetection\20221025-Processed-EATD-feature\extractors")
sys.path.append(r"D:\[SYNC]存档\[2023.03.01]DepressionDetection\20221025-Processed-EATD-feature\models")
sys.path.append(r"F:\ffmpeg\bin")

print(sys.path)

import os

from main import FeatureExtractionTool
# from MSA_FET import FeatureExtractionTool


assert os.path.exists("./EATD-Corpus/t_1/positive.wav")
assert os.path.exists("./EATD-Corpus/t_1/negative.wav")
assert os.path.exists("./EATD-Corpus/t_1/neutral.wav")
# assert os.path.exists("./SIMS-Corpus/0001.mp4")

EXTRACT_AUDIO = True

EXTRACT_TEXT = False

EXTRACT_AUXILIARY = False

EXTRACT_HUBERT = False

EXTRACT_GPT3 = False

EXTRACT_EMOTION_KNOWLEDGE = False

EXTRACT_EMOTION_EMBS = False


def azure_translator(text):
    import requests, uuid, json

    # Add your key and endpoint
    key = "1870e4e8a2ae49308a8a00b05683750b"
    endpoint = "https://api.cognitive.microsofttranslator.com"

    # location, also known as region.
    # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
    location = "eastasia"

    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'zh-Hans',
        'to': 'en'
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': text
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    translate_text = response[0]['translations'][0]['text']

    return translate_text
    # print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))


if EXTRACT_AUDIO:
    # fet = FeatureExtractionTool("librosa", tmp_dir='./tmp', log_dir='./log')
    fet = FeatureExtractionTool("opensmile", tmp_dir='./tmp', log_dir='./log')

    # pos_feature = fet.run_single("./EATD-Corpus/t_1/positive.wav")
    # neu_feature = fet.run_single("./EATD-Corpus/t_1/neutral.wav")
    # neg_feature = fet.run_single("./EATD-Corpus/t_1/negative.wav")
    #
    # print('pos_feature',pos_feature['audio'].shape)
    # print('neu_feature',neu_feature['audio'].shape)
    # print('neg_feature',neg_feature['audio'].shape)
    # feature = fet.run_single("./SIMS-Corpus/0001.mp4")

    # print(feature)

    # extract EATD dataset


    polarities = ['negative', 'positive', 'neutral']

    max_index = 200
    eatd_path = './EATD-Corpus'



    def extract_feature(prefix):

        feature_lists = []
        length_list = []

        for i in range(max_index):
            # if i < 70:
            #     continue
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                feature_per = []
                for p_idx, polarity in enumerate(polarities):
                    # wav_file = f'{dir_path}/{polarity}_out.wav'
                    wav_file = f'{dir_path}/{polarity}_synthetic.wav' # 针对合成文本
                    wav_feature = np.nan_to_num(fet.run_single(wav_file)['audio'])
                    print(f'{polarity} shape', wav_feature.shape)


                    feature_per.append(wav_feature)
                    length_list.append(wav_feature.shape[0])
                feature_lists.append(feature_per)

            else:
                print(f'pass {prefix}_{i}')
                continue

        return feature_lists, length_list

    def pad_audio_feature(feature_list, max_len):
        padded_feature_list = []

        for pnn_feature in feature_list:
            padded_pnn_list = []
            for feature in pnn_feature:
                padded_feature = np.pad(feature, ((0, (max_len- feature.shape[0])), (0, 0)), mode="constant")
                padded_pnn_list.append(padded_feature)


            padded_feature_list.append(np.array(padded_pnn_list))

        padded_feature = np.array(padded_feature_list)

        return padded_feature




    train_feature_list, train_length_list = extract_feature('t')
    valid_feature_list, valid_length_list = extract_feature('v')

    max_len = max(train_length_list + valid_length_list)

    train_features = pad_audio_feature(train_feature_list, max_len)
    valid_features = pad_audio_feature(valid_feature_list, max_len)

    print('train_features', train_features.shape)
    print('valid_features', valid_features.shape)


    # np.savez(os.path.join('EATD-Features/train_audio_new_features_b_3_len_88.npz'), train_features)
    # np.savez(os.path.join('EATD-Features/valid_audio_new_features_b_3_len_88.npz'), valid_features)
    #
    # np.savez(os.path.join('EATD-Features/train_synaudio_opensmile_88.npz'), train_features) # 0622
    # np.savez(os.path.join('EATD-Features/valid_synaudio_opensmile_88.npz'), valid_features)



if EXTRACT_TEXT:
    fet = FeatureExtractionTool(config="bert", tmp_dir='./tmp', log_dir='./log')

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
                    # text_file = f'{dir_path}/{polarity}.txt' # 针对原始文本
                    # text_file = f'{dir_path}/{polarity}_synthetic.txt' # 针对合成文本
                    text_file = f'{dir_path}/{polarity}_synthetic_v2.txt' # 针对合成文本
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

    # np.savez(os.path.join('EATD-Features/train_text_features_b_3_len_768.npz'), train_features)
    # np.savez(os.path.join('EATD-Features/valid_text_features_b_3_len_768.npz'), valid_features)

    # np.save('EATD-Features/train_text_simple_roberta_768.npy', train_features) # 针对原始文本
    # np.save('EATD-Features/valid_text_simple_roberta_768.npy', valid_features)

    # np.save('EATD-Features/train_syntext_simple_roberta_768.npy', train_features) # 针对合成文本
    # np.save('EATD-Features/valid_syntext_simple_roberta_768.npy', valid_features)

    np.save('EATD-Features/train_syntext_v2_simple_roberta_768.npy', train_features) # 针对合成文本
    np.save('EATD-Features/valid_syntext_v2_simple_roberta_768.npy', valid_features)


if EXTRACT_AUXILIARY:

    'questions'

    fet = FeatureExtractionTool(config="bert", tmp_dir='./tmp', log_dir='./log')

    tasks = ['gender', 'questions']
    polarities = ['negative', 'positive', 'neutral']

    max_index = 200
    eatd_path = './EATD-Corpus'

    def extract_question_feature(prefix):

        feature_lists = []
        length_list = []

        for i in range(max_index):
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                # questions_f = f'{dir_path}/neg_pos_neu_questions.txt'
                # for q_idx, question in enumerate(open(questions_f, 'r', encoding='utf-8').readlines()):
                #     with open(f'{dir_path}/{polarities[q_idx]}_question.txt', 'w', encoding='utf-8') as f:
                #         f.write(question)

                feature_per = []
                for p_idx, polarity in enumerate(polarities):
                    text_file = f'{dir_path}/{polarity}_question.txt'
                    text_feature = (fet.run_single(None, text_file=text_file)['text'])
                    print(f'{polarity} question shape', text_feature.shape)

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

    # train_question_list, train_length_list = extract_question_feature('t')
    # valid_question_list, valid_length_list = extract_question_feature('v')
    # max_len = max(train_length_list + valid_length_list)
    #
    # train_features = pad_text_feature(train_question_list, max_len)
    # valid_features = pad_text_feature(valid_question_list, max_len)
    # #
    # print('train_features', train_features.shape)
    # print('valid_features', valid_features.shape)
    # #
    # np.savez(os.path.join('EATD-Features/train_questions_b_3_len_768.npz'), train_features)
    # np.savez(os.path.join('EATD-Features/valid_questions_b_3_len_768.npz'), valid_features)



    'gender'

    def extract_gender_feature(prefix):

        gender_list = []

        for i in range(max_index):
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                gender = open(f'{dir_path}/gender.txt', 'r', encoding='utf-8').readlines()[0].strip()

                if gender == 'm':
                    gender_list.append(1.)
                elif gender == 'f':
                    gender_list.append(0.)
                else:
                    raise ValueError

            else:
                print(f'pass {prefix}_{i}')
                continue

        return gender_list

    train_gender = extract_gender_feature('t')
    valid_gender = extract_gender_feature('v')
    #
    np.savez(os.path.join('EATD-Features/train_gender_b.npz'), np.array(train_gender))
    np.savez(os.path.join('EATD-Features/valid_gender_b.npz'), np.array(valid_gender))

if EXTRACT_HUBERT:
    from models.HuBERT_segment_extractor import HuBERT_segment_extractor

    hubert = HuBERT_segment_extractor()

    polarities = ['negative', 'positive', 'neutral']

    max_index = 200
    eatd_path = './EATD-Corpus'


    def extract_feature(prefix):

        feature_lists = []
        length_list = []

        for i in range(max_index):
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                print(f'Process {prefix}_{i}')

                feature_per = []
                for p_idx, polarity in enumerate(polarities):
                    wav_file = f'{dir_path}/{polarity}_out.wav'
                    wav_feature = np.nan_to_num(hubert.fet_extract(wav_file))
                    print(f'{polarity} shape', wav_feature.shape)

                    feature_per.append(wav_feature)
                    length_list.append(wav_feature.shape[0])
                feature_lists.append(feature_per)

            else:
                print(f'Pass {prefix}_{i}')
                continue
            # if i > 10:
            #     break

        return feature_lists, length_list


    def pad_audio_feature(feature_list, max_len):
        padded_feature_list = []

        for pnn_feature in feature_list:
            padded_pnn_list = []
            for feature in pnn_feature:
                padded_feature = np.pad(feature, ((0, (max_len - feature.shape[0])), (0, 0)), mode="constant")
                padded_pnn_list.append(padded_feature)

            padded_feature_list.append(np.array(padded_pnn_list))

        padded_feature = np.array(padded_feature_list)

        return padded_feature


    train_feature_list, train_length_list = extract_feature('t')
    valid_feature_list, valid_length_list = extract_feature('v')

    max_len = max(train_length_list + valid_length_list)

    train_features = pad_audio_feature(train_feature_list, max_len)
    valid_features = pad_audio_feature(valid_feature_list, max_len)

    print('train_features', train_features.shape)
    print('valid_features', valid_features.shape)

    np.savez(os.path.join('EATD-Features/train_audio_hubert_features_b_3_len_768.npz'), train_features)
    np.savez(os.path.join('EATD-Features/valid_audio_hubert_features_b_3_len_768.npz'), valid_features)


if EXTRACT_GPT3:
    '''
    输入： 样本的性别、问题
    输出： 1. 音频标准回答 b, 3, 2, len, 88 2. 文本标准回答 b, 3, 2, len, 768
    '''
    question_dict = {'最讨厌什么样的人？': 0,
                     '放松的时候喜欢干什么？': 1,
                     '希望从事什么工作？': 2,
                     '最后悔的事情是什么？': 3,
                     '你喜欢什么样的工作环境？': 4,
                     '最喜欢吃什么食物？': 5,
                     '你的家庭构成是怎么样的？': 6,
                     '上一次和别人吵架是什么时候？': 7,
                     '你对小孩子有什么看法？': 8,
                     '最近一件开心的事是什么？': 9,
                     '你最好的朋友是谁？': 10,
                     '你最喜欢家乡的哪方面？': 11,
                     '大学学习什么专业？': 12,
                     '印象最深的一件事是什么？': 13,
                     '最近有失眠吗？': 14,
                     '生活中有哪些人给你带来了正面的影响？': 15,
                     '独处的时候一般做什么？': 16,
                     '最近一件不开心的事是什么？': 17,
                     '生气的时候会如何处理？': 18,
                     '最讨厌的事情是什么？': 19,
                     '你男/女朋友怎么样？': 20,
                     '你父母是什么样的人？': 21
                     }

    gender_dict = { 'm': 'male',
                    'f': 'female'}

    max_index = 200
    eatd_path = './EATD-Corpus'
    # feature_type = 'text'
    feature_type = 'audio'


    def extract_feature(prefix, feature_type):

        print(f'Get GPT3 {feature_type} features .')
        male_healthy_features = np.load(f'./GPT3-Response/EATD/{feature_type}_male_healthy_features.npy')
        male_depression_features = np.load(f'./GPT3-Response/EATD/{feature_type}_male_depression_features.npy')
        female_healthy_features = np.load(f'./GPT3-Response/EATD/{feature_type}_female_healthy_features.npy')
        female_depression_features = np.load(f'./GPT3-Response/EATD/{feature_type}_female_depression_features.npy')



        feature_list = []

        for i in range(max_index):
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                print(f'Process {prefix}_{i}')

                with open(f'{dir_path}/gender.txt', 'r', encoding='utf-8') as f:
                    gender = f.readlines()[0].strip()

                question_list = []

                with open(f'{dir_path}/neg_pos_neu_questions.txt', 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        question_list.append(question_dict[line.strip()])


                if gender == 'm':
                    healthy_features = male_healthy_features
                    depression_features = male_depression_features

                elif gender == 'f':
                    healthy_features = female_healthy_features
                    depression_features = female_depression_features
                else:
                    raise ValueError

                feature_per = []
                for question in question_list:
                    healthy_feature = healthy_features[question, :, :] # len, feature
                    depression_feature = depression_features[question, :, :]

                    feature_per.append([healthy_feature, depression_feature])

                feature_list.append(feature_per)


            else:
                print(f'Pass {prefix}_{i}')
                continue
            # if i > 10:
            #     break

        features = np.array(feature_list) # b, 3, 2, len, dim
        return features


    train_features_gpt3 = extract_feature('t', feature_type)
    valid_features_gpt3 = extract_feature('v', feature_type)


    print('train_features', train_features_gpt3.shape)
    print('valid_features', valid_features_gpt3.shape)

    np.savez(os.path.join(f'EATD-Features/train_{feature_type}_gpt3_features_b_3_2_len_768.npz'), train_features_gpt3)
    np.savez(os.path.join(f'EATD-Features/valid_{feature_type}_gpt3_features_b_3_2_len_768.npz'), valid_features_gpt3)

if EXTRACT_EMOTION_KNOWLEDGE:

    import torch
    from torch.nn import functional as F
    from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

    tokenizer = RobertaTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
    model = RobertaForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

    # fet = FeatureExtractionTool(config="bert", tmp_dir='./tmp', log_dir='./log')

    polarities = ['negative', 'positive', 'neutral']

    max_index = 200
    eatd_path = './EATD-Corpus'

    emotion_dict = {
     0: 'anger',
     1: 'disgust',
     2: 'fear',
     3: 'joy',
     4: 'neutral',
     5: 'sadness',
     6: 'surprise',
    }

    def extract_text_feature(prefix):

        feature_list = []
        prob_list = []

        for i in range(max_index):
            # if i > 3:
            #     break

            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                feature_per = []
                prob_per = []
                for p_idx, polarity in enumerate(polarities):
                    text_file = f'{dir_path}/{polarity}.txt'
                    with open(text_file, 'r', encoding='utf-8') as f:
                        answer_zh = f.readlines()[0].strip()

                        answer_en = azure_translator(answer_zh)

                        print(f'中文回答：{answer_zh}')
                        print(f'英文回答：{answer_en}')

                        encoded_input = tokenizer(answer_en, return_tensors='pt')
                        with torch.no_grad():
                            roberta_feature = model(**encoded_input, output_hidden_states=True).hidden_states[-1][0, 0, :]
                            roberta_logits = model(**encoded_input, output_hidden_states=True).logits
                            roberta_pred = F.softmax(roberta_logits, -1)[0, :]

                            roberta_feature = roberta_feature.numpy()
                            roberta_pred = roberta_pred.numpy()
                            print(f'情绪预测：{emotion_dict[int(np.argmax(roberta_pred))]}')



                        pass



                    # print(f'roberta_feature shape', roberta_feature.shape)
                    # print(f'roberta_pred shape', roberta_pred.shape)
                    print('\n')

                    feature_per.append(roberta_feature)
                    prob_per.append(roberta_pred)
                feature_list.append(feature_per)
                prob_list.append(prob_per)

            else:
                print(f'pass {prefix}_{i}')
                continue

        return np.array(feature_list), np.array(prob_list)


    train_emotion_feature, train_emotion_prob = extract_text_feature('t')
    valid_emotion_feature, valid_emotion_prob = extract_text_feature('v')

    print('train_emotion_feature', train_emotion_feature.shape)
    print('train_emotion_prob', train_emotion_prob.shape)

    print('valid_emotion_feature', valid_emotion_feature.shape)
    print('valid_emotion_prob', valid_emotion_prob.shape)

    np.savez(os.path.join('EATD-Features/train_emotion_features_b_3_768.npz'), train_emotion_feature)
    np.savez(os.path.join('EATD-Features/valid_emotion_features_b_3_768.npz'), valid_emotion_feature)

    np.savez(os.path.join('EATD-Features/train_emotion_probs_b_3_7.npz'), train_emotion_prob)
    np.savez(os.path.join('EATD-Features/valid_emotion_probs_b_3_7.npz'), valid_emotion_prob)

if EXTRACT_EMOTION_EMBS:

    import torch
    from torch.nn import functional as F
    from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

    tokenizer = RobertaTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
    model = RobertaForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

    # fet = FeatureExtractionTool(config="bert", tmp_dir='./tmp', log_dir='./log')

    polarities = ['negative', 'positive', 'neutral']

    max_index = 200
    eatd_path = './EATD-Corpus'

    emotion_dict = {
     0: 'anger',
     1: 'disgust',
     2: 'fear',
     3: 'joy',
     4: 'neutral',
     5: 'sadness',
     6: 'surprise',
    }

    def extract_text_feature(prefix):

        feature_lists = []
        length_list = []

        for i in range(max_index):
            # if i > 10:
            #     break
            dir_path = f'{eatd_path}/{prefix}_{i}'
            if os.path.exists(dir_path):
                feature_per = []
                for p_idx, polarity in enumerate(polarities):
                    text_file = f'{dir_path}/{polarity}.txt'
                    # text_feature = (fet.run_single(None, text_file=text_file)['text'])

                    with open(text_file, 'r', encoding='utf-8') as f:
                        answer_zh = f.readlines()[0].strip()

                        answer_en = azure_translator(answer_zh)

                        print(f'中文回答：{answer_zh}')
                        print(f'英文回答：{answer_en}')

                        encoded_input = tokenizer(answer_en, return_tensors='pt')
                        with torch.no_grad():
                            roberta_feature = model(**encoded_input, output_hidden_states=True).hidden_states[-1][0, :, :]
                            roberta_logits = model(**encoded_input, output_hidden_states=True).logits
                            roberta_pred = F.softmax(roberta_logits, -1)[0, :]

                            roberta_feature = roberta_feature.numpy()
                            roberta_pred = roberta_pred.numpy()
                            print(f'情绪预测：{emotion_dict[int(np.argmax(roberta_pred))]}')


                    print(f'{polarity} shape', roberta_feature.shape)

                    feature_per.append(roberta_feature)
                    length_list.append(roberta_feature.shape[0])
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

    np.savez(os.path.join('EATD-Features/train_emotion_embs_b_3_len_768.npz'), train_features)
    np.savez(os.path.join('EATD-Features/valid_emotion_embs_b_3_len_768.npz'), valid_features)

