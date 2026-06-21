import time
import os
SEED = 123
import numpy as np
from collections import Counter
from numpy.random import seed
import pandas as pd
seed(SEED)

# GLOVE_PATH = r'X:\数据集+预训练向量\glove.840B.300d.txt'
GLOVE_PATH = r'F:\预训练模型及词向量\GloVe通用词向量\glove.840B.300d.txt'

def data_init(input_path, output_path):
    source_count = []
    source_word2idx = {}
    max_sent_len = 0



    for folder in os.listdir(input_path):
        folder_path = input_path + folder
        if os.path.isdir(folder_path):

            source_words = []

            question_path = os.path.join(folder_path, f"questions.txt")
            answer_path = os.path.join(folder_path, f"answers.txt")
            synthetic_path = os.path.join(folder_path, f"syn_answers.txt")

            with open(question_path, 'r', encoding='utf-8') as f:
                question_lines = f.readlines()
            with open(answer_path, 'r', encoding='utf-8') as f:
                answer_lines = f.readlines()

            if os.path.exists(synthetic_path):
                print(f'Processing synthetic answers: {synthetic_path}')
                with open(synthetic_path, 'r', encoding='utf-8') as f:
                    synthetic_lines = f.readlines()
            else:
                synthetic_lines = []

            text_lines = question_lines + answer_lines + synthetic_lines

            for line in text_lines:
                sptoks = line.replace('.', '').strip().split()
                source_words.extend([sp.lower() for sp in sptoks])
                if len(sptoks) > max_sent_len:
                    max_sent_len = len(sptoks)

            if len(source_count) == 0:
                source_count.append(['<pad>', 0])
            source_count.extend(Counter(source_words).most_common())
            for word, _ in source_count:
                if word not in source_word2idx:
                    source_word2idx[word] = len(source_word2idx)

    # splits = ['train', 'dev']
    #
    # for split in splits:
    #     source_words = []
    #     print('Processing {}...'.format(split))
    #
    #     dataframe = pd.read_csv(f'./{split}_split_Depression_AVEC2017.csv')
    #
    #     patient_total_num, annotation_total_num = dataframe.shape
    #
    #     feature_list = []
    #     qa_num_list = []
    #
    #     for patient_num, patient_info in enumerate(dataframe.itertuples()):
    #         feature_per = []
    #
    #         patient_id = getattr(patient_info, 'Participant_ID')
    #         PREFIX = f'../SimpleProcessed/{split}/{patient_id}'
    #         with open(f'{PREFIX}/{patient_id}_transcript.txt', 'r') as f:
    #
    #             lines = f.readlines()
    #             response_cnt = 0
    #             responses = []
    #             for line in lines:
    #                 if line.startswith('RESPONSE'):
    #                     response_cnt += 1
    #                     # print('LINE', line)
    #                     response = line.strip().split('\t')[1]
    #
    #                     sptoks = response.strip().split()
    #                     source_words.extend([sp.lower() for sp in sptoks])
    #                     if len(sptoks) > max_sent_len:
    #                         max_sent_len = len(sptoks)
    #
    #     if len(source_count) == 0:
    #         source_count.append(['<pad>', 0])
    #     source_count.extend(Counter(source_words).most_common())
    #     for word, _ in source_count:
    #         if word not in source_word2idx:
    #             source_word2idx[word] = len(source_word2idx)

    # print(source_count)
    # print(source_word2idx)
    print('max_sentence_length', max_sent_len)

    with open(output_path+'word2id.txt', 'w', encoding='utf-8') as f:
        f.write(str(source_word2idx))

    return source_word2idx

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def init_word_embeddings(path, word2idx, embedding_file, oovname, dimension):
    print('path', path)
    wt = np.zeros([len(word2idx), dimension])
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
    cnt = 0

    with open(path + oovname, "w", encoding='utf-8') as f:
        for w in word2idx:
            if w == '<pad>':
                continue
            if np.sum(wt[word2idx[w]]) == 0.:
                cnt += 1
                f.write(w+"\n")

    # print('HIT rate: {:.2f}%, OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))
    return wt

def compute_ngrams(word, min_n, max_n):
    BOW, EOW = ('', '')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = BOW + word + EOW
    ngrams = []
    if 2 < len(word) <= 4:
        min_n = 2
    elif len(word) <= 2:
        min_n = 1
        pass

    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams

def get_oov_vector(path, oov_dict, source_emb, oov_emb, dimension):
    word2ngram_dict = {}
    ngram2idx_dict = {}
    with open(path + oov_dict, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            ngrams = compute_ngrams(word, 3, 10)
            # print(word, ngrams)
            word2ngram_dict[word] = ngrams
            for ngram in ngrams:
                if ngram not in ngram2idx_dict:
                    ngram2idx_dict[ngram] = len(ngram2idx_dict)

    ngram_vec = np.zeros([len(ngram2idx_dict), dimension])
    with open(source_emb, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in ngram2idx_dict:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                ngram_vec[ngram2idx_dict[content[0]]] = np.array(list(map(np.float32, content[1:])))


    # print(word2ngram_dict)
    # print(ngram2idx_dict)
    # print(ngram_vec)

    word2vec_dict = {}
    for word in word2ngram_dict:
        word_vec = np.zeros([dimension])
        ngram_cnt = 0
        ngrams = word2ngram_dict[word]
        for ngram in ngrams:
            if np.sum(ngram_vec[ngram2idx_dict[ngram]]) != 0:
                word_vec += ngram_vec[ngram2idx_dict[ngram]]
                ngram_cnt += 1
        word_vec /= float(ngram_cnt)+1e-6
        word2vec_dict[word] = word_vec

    cnt = 0
    for word in word2vec_dict:
        if np.sum(word2vec_dict[word]) == 0.:
            cnt += 1
    print('Ngram HIT rate: {:.2f}%, Ngram OOV rate: {:.2f}%'.format(100 - cnt/len(word2vec_dict) * 100, cnt/len(word2vec_dict) * 100))

    with open(path + oov_emb, 'w', encoding='utf-8') as f:
        for word in word2vec_dict:
            f.write(word + ' ')
            for number in word2vec_dict[word]:
                f.write(str(number) + ' ')
            f.write('\n')

def full_embedding(path, word2idx, iv_txt, oov_txt, outemb, dimension):
    # print('path', path)
    wt = np.zeros([len(word2idx), dimension])

    'IN VOCABULARY'
    with open(iv_txt, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
    cnt = 0
    for w in word2idx:
        if w == '<pad>':
            continue
        if np.sum(wt[word2idx[w]]) == 0.:
            cnt += 1
    print('Raw   HIT rate: {:.2f}%, Raw   OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))

    'OUT OF VOCABULARY'
    with open(path + oov_txt, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                if np.sum(wt[word2idx[content[0]]]) == 0.:
                    wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
                else:
                    raise ValueError

    cnt = 0
    for w in word2idx:
        if w == '<pad>':
            continue
        if np.sum(wt[word2idx[w]]) == 0.:
            cnt += 1
    print('Plus  HIT rate: {:.2f}%, Plus  OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))

    'RANDOM INITIALIZE'
    for idx in range(len(wt)):
        if idx == 0:
            continue
        else:
            if np.sum(wt[idx]) == 0.:
                wt[idx] = np.random.normal(-0.05, 0.05, [dimension])
            else:
                pass

    cnt = 0
    for w in word2idx:
        if w == '<pad>':
            continue
        if np.sum(wt[word2idx[w]]) == 0.:
            cnt += 1
    print('Final HIT rate: {:.2f}%, Final OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))

    wt = np.asarray(wt, dtype=np.float32)
    wt[0,:] = 0.0
    np.save(path + outemb, wt)
    return wt

# For generating the word-idx mapping and the word vectors,
# place the glove embedding and the corresponding domain embedding in the folder of each dataset.

input_path = '../Embedding/'
output_path = './Glove_Preprocess/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

print('Initialize Word Dictionary & Embedding')
word_dict = data_init(input_path, output_path)

init_word_embeddings(output_path, word_dict, GLOVE_PATH, 'glove_word_oov.txt', 300)
get_oov_vector(output_path, 'glove_word_oov.txt', GLOVE_PATH, 'glove_emb_oov.txt', 300)
full_embedding(output_path, word_dict, GLOVE_PATH, 'glove_emb_oov.txt', 'glove_embedding.npy', 300)






