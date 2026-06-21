import os
import sys

import numpy as np
import winsound

polarities = ['negative', 'positive', 'neutral']

max_index = 200
eatd_path = './EATD-Corpus'


def annotate_auxiliary_info(prefix):

    feature_lists = []
    length_list = []

    for i in range(max_index):
        dir_path = f'{eatd_path}/{prefix}_{i}'
        if os.path.exists(dir_path):

            annotate_f = open(f'{dir_path}/neg_pos_neu_questions.txt', 'w', encoding='utf-8')

            for p_idx, polarity in enumerate(polarities):
                text_file = f'{dir_path}/{polarity}.txt'
                print('Answer:', open(text_file, 'r', encoding='utf-8').readlines()[0].strip())
                question = input("Input correspondings question:")
                annotate_f.write(question)
                annotate_f.write('\n')
            annotate_f.close()

        else:
            print(f'pass {prefix}_{i}')
            continue

    return feature_lists, length_list


def annotate_gender_info(prefix):

    feature_lists = []
    length_list = []

    for i in range(max_index):
        dir_path = f'{eatd_path}/{prefix}_{i}'
        if os.path.exists(dir_path):

            # annotate_f = open(f'{dir_path}/gender.txt', 'w', encoding='utf-8')
            annotate_f = open(f'{dir_path}/gender.txt', 'r', encoding='utf-8').readlines()[0]

            audio_file = f'{dir_path}/negative_out.wav'
            winsound.PlaySound(audio_file, winsound.SND_FILENAME|winsound.SND_ASYNC)

            print(annotate_f)
            gender = input("Check User Gender:")

            # annotate_f.write(gender)
            # annotate_f.write('\n')
            #
            # annotate_f.close()

        else:
            print(f'pass {prefix}_{i}')
            continue

    return feature_lists, length_list



# train_feature_list, train_length_list = annotate_auxiliary_info('t')
# valid_feature_list, valid_length_list = annotate_auxiliary_info('v')

train_feature_list, train_length_list = annotate_gender_info('t')
valid_feature_list, valid_length_list = annotate_gender_info('v')