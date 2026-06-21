import csv
import os
import pandas as pd
import numpy as np
import struct
import spacy

nlp = spacy.load('en_core_web_lg')


# Traverse all CSV files in the "Corpus" folder
corpus_path = '../Corpus'


def merge_multiple_utterance(corpus_path):
    def merge_utterances(filename, data):
        merged_data = []
        current_speaker = None
        current_start_time = None
        current_stop_time = None
        current_utterance = []

        for row in data:
            try:
                start_time, stop_time, speaker, value = row
            except ValueError:
                print(f'Decode error in {filename}. Line: {row}')
                pass

            if speaker == current_speaker:
                # Same speaker, extend the current utterance
                current_stop_time = stop_time
                current_utterance.append(value)
            else:
                # Different speaker, save the current utterance and start a new one
                if current_speaker is not None:
                    merged_data.append(
                        (current_start_time, current_stop_time, current_speaker, '. '.join(current_utterance) + '.'))

                current_speaker = speaker
                current_start_time = start_time
                current_stop_time = stop_time
                current_utterance = [value]

        # Don't forget to save the last utterance
        if current_speaker is not None:
            merged_data.append(
                (current_start_time, current_stop_time, current_speaker, '. '.join(current_utterance) + '.'))

        return merged_data

    corpus_path = f"{corpus_path}/Raw Transcript"
    for filename in os.listdir(corpus_path):
        if filename.endswith('.csv') and "merged" not in filename:
        # if filename.endswith('.csv') and "merged" not in filename and "REVISE" in filename and filename[7:10] in ["451", "480"]:
            # Read the CSV data
            with open(os.path.join(corpus_path, filename), 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # Skip the header
                data = list(reader)

            merged_data = merge_utterances(filename, data)

            # Save the merged data to a new CSV file
            with open(os.path.join(corpus_path, 'merged_' + filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['start_time', 'stop_time', 'speaker', 'value'])  # Write the header
                writer.writerows(merged_data)

# merge_multiple_utterance(corpus_path)

def get_question_answer_embedding(corpus_path):
    import os
    import numpy as np
    import pandas as pd
    import spacy
    import re

    nlp = spacy.load('en_core_web_lg')

    # function to calculate average vector of a sentence
    def avg_sentence_vector(sentence):
        doc = nlp(sentence)
        return np.mean(np.array([token.vector for token in doc]), axis=0)

    text_dir = f"{corpus_path}/Merged Transcript"
    for file in os.listdir(text_dir):
        if file.endswith('.csv') and "merged" in file:
        # if file.endswith('.csv') and "merged" in file and file[7:10] in ["451", "480"]:
            print(f"Processing: {file}")

            # folder_name = file[:-4]
            # if not os.path.exists(folder_name):
            #     os.makedirs(folder_name)

            df = pd.read_csv(os.path.join(text_dir, file))
            questions = []
            answers = []


            for i in range(len(df) - 1):
                if df.iloc[i]['speaker'] == 'Ellie' and df.iloc[i + 1]['speaker'] == 'Participant':
                    q = df.iloc[i]['value']
                    a = df.iloc[i + 1]['value']
                    # pair_num = i // 2 + 1

                    # Segment the text with Spacy
                    q_doc = nlp(q)
                    a_doc = nlp(a)

                    # Check the word count
                    if len(q_doc) >= 3 or len(a_doc) >= 3:

                    # if len(q.split()) >= 3 or len(a.split()) >= 3:
                        # with open(f'{folder_name}/question_{pair_num}.txt', 'w') as f:
                        #     f.write(q)
                        # with open(f'{folder_name}/answer_{pair_num}.txt', 'w') as f:
                        #     f.write(a)

                        if '(' in q and ')' in q:  # 遇到Ellie缩写的情况，直接用括号内的自然语言问题
                            # print(question)
                            # print(patient_id)
                            q = re.findall(r"[(](.*?)[)]", q)[0]
                            print(q)

                        questions.append(q)
                        answers.append(a)



            q_vecs = [avg_sentence_vector(q) for q in questions]
            a_vecs = [avg_sentence_vector(a) for a in answers]

            embedding_dir = f'{os.path.splitext(file)[0]}_embedding'
            os.makedirs(embedding_dir, exist_ok=True)

            np.save(os.path.join(embedding_dir, 'ellie_emb.npy'), np.array(q_vecs))
            np.save(os.path.join(embedding_dir, 'participant_emb.npy'), np.array(a_vecs))

            with open(f'{embedding_dir}/questions.txt', 'w') as f:
                f.write('\n'.join(questions))
            with open(f'{embedding_dir}/answers.txt', 'w') as f:
                f.write('\n'.join(answers))

# get_question_answer_embedding(corpus_path)

def get_audio_split(corpus_path):
    import pandas as pd
    import spacy
    from pydub import AudioSegment
    import os
    import re

    nlp = spacy.load("en_core_web_lg")

    # Read the CSV file
    text_dir = f"{corpus_path}/Merged Transcript"
    audio_dir = f"{corpus_path}/Raw Audio"

    for file in os.listdir(text_dir):
        if file.endswith('.csv') and "merged" in file:
        # if file.endswith('.csv') and "merged" in file and file[7:10] in ["451", "480"]:
            print(f"Processing: {file}")

            csv_filename = os.path.join(text_dir, file)

            data = pd.read_csv(csv_filename)

            # Create the X_Split_Audio directory if it doesn't exist
            match = re.search(r'\d+', file)
            participant_name = match.group()

            folder_name = f"{participant_name}_Split_Audio"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Load the audio file
            wav_path = f"{audio_dir}/{participant_name}_AUDIO.wav"
            audio = AudioSegment.from_wav(wav_path)

            pair_num = 0
            pairs = []
            for i in range(len(data) - 1):
                if data.iloc[i]['speaker'] == 'Ellie' and data.iloc[i + 1]['speaker'] == 'Participant':
                    q = data.iloc[i]['value']
                    a = data.iloc[i + 1]['value']

                    q_doc = nlp(q)
                    a_doc = nlp(a)

                    if len(q_doc) >= 3 or len(a_doc) >= 3:
                        pair_num += 1
                        start_time = int(float(data.iloc[i + 1]['start_time']) * 1000)  # Convert to milliseconds
                        stop_time = int(float(data.iloc[i + 1]['stop_time']) * 1000)
                        clip = audio[start_time:stop_time]

                        # Save the audio clip
                        clip.export(f"{folder_name}/{participant_name}_participant_{pair_num}.wav", format="wav")

# get_audio_split(corpus_path)

def get_audio_embedding(corpus_path):
    import opensmile
    import soundfile as sf
    import numpy as np
    import os
    from scipy import signal

    # Prepare the feature extractor
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        options={'window': 0.06, 'step': 0.01}  # Frame size and step in seconds
    )

    root_dir = r'D:\[SYNC]代码\Coding\[2023.06.13]Prompt-DAICWOZ\Corpus\Split Audio'
    for folder in os.listdir(root_dir):
        if folder.endswith('_Split_Audio'):
            if int(folder[:3]) < 480 and int(folder[:3]) != 451:
                continue
            print(f"Processing folder {folder}")
            qa_pair_features = []
            for i in range(1, 1000):  # go through all questions and answers
                segment_features = []
                file_path = os.path.join(root_dir, folder, f'{folder[:-12]}_participant_{i}.wav')
                if not os.path.exists(file_path):
                    print(f"Folder {folder} QA pairs break at {i}")
                    break

                print(f"Processing folder {folder} QA pair {i}")

                # Load the audio file
                audio, samplerate = sf.read(file_path)

                # Segment the audio sequence into one-second segments
                segments = []
                segment_length = samplerate  # One second
                for start in range(0, len(audio), segment_length):
                    end = start + segment_length
                    segment = audio[start:end]
                    segments.append(segment)

                for segment in segments:
                    # Zero pad the segment if it's shorter than one second
                    if len(segment) < segment_length:
                        segment = np.pad(segment, (0, segment_length - len(segment)))

                    # Extract features from the segment
                    segment_feature = smile.process_signal(segment, samplerate).squeeze(0)  # 88-dim

                    # Append the features of the current segment to the features list
                    segment_features.append(segment_feature)  # seconds *  (1, 88)

                qa_pair_feature = np.mean(segment_features, axis=0, keepdims=True)  # (1, 88)
                qa_pair_features.append(qa_pair_feature)  # 200 * (1, 88) QA num
            # Average the features
            audio_feature = np.concatenate(qa_pair_features, axis=0)  #

            # Save the average features
            np.save(f'../Embedding/merged_{folder[:-12]}_TRANSCRIPT_embedding/audio_emb.npy', audio_feature)

    pass

# get_audio_embedding(corpus_path)

def get_clnf_hog_features():
    def read_clnf_hog_bin(file_name):
        hog_vectors = []

        with open(file_name, "rb") as f:
            while True:
                header = f.read(4*4)  # read num_cols, num_rows, num_channels, valid_frame
                if not header:  # break the loop if we've reached the end of the file
                    break
                num_cols, num_rows, num_channels, valid_frame = struct.unpack('iiii', header)
                if valid_frame:
                    hog_vector = struct.unpack('f'*4464, f.read(4464*4))  # read and unpack the 4464d vector
                    hog_vectors.append(hog_vector)

        return np.array(hog_vectors)

    def extract_clnf_hog_features(file_name, transcript_file):
        hog_data = read_clnf_hog_bin(file_name)
        time_values = np.arange(0, len(hog_data) / 30, 1 / 30)

        df = pd.read_csv(transcript_file)
        # df = df[df['speaker'] == 'Participant'] # Filter out only the Participant's utterances

        utterance_features = []
        for i in range(len(df) - 1): # Ensure that only extract QA pairs
            if df.iloc[i]['speaker'] == 'Ellie' and df.iloc[i + 1]['speaker'] == 'Participant':
                q = df.iloc[i]['value']
                a = df.iloc[i + 1]['value']
                # pair_num = i // 2 + 1

                # Segment the text with Spacy
                q_doc = nlp(q)
                a_doc = nlp(a)

                # Check the word count
                if len(q_doc) >= 3 or len(a_doc) >= 3:
                    start_time, stop_time = df.iloc[i + 1]['start_time'], df.iloc[i + 1]['stop_time']

                    # Add a 1/30 second buffer to the start and stop times
                    start_time = start_time - 1/30
                    stop_time = stop_time + 1/30

                    indices = np.where((time_values >= start_time) & (time_values <= stop_time))[0]
                    utterance_hog = np.mean(hog_data[indices], axis=0)
                    utterance_hog = np.nan_to_num(utterance_hog)
                    utterance_features.append(utterance_hog)
        return np.array(utterance_features)


    dataset_path = r"F:\数据集收集\抑郁检测-DAIC-WOZ\189SAMPLES"
    merged_path = r"D:\[SYNC]代码\Coding\[2023.06.13]Prompt-DAICWOZ\Corpus\Merged Transcript"

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
        # if os.path.isdir(folder_path) and "402" in folder_path:
            index = int(folder[:3])
            print(f"Processing {folder} HOG feature...")
            hog_file_path = os.path.join(folder_path, f"{index}_CLNF_hog.bin")
            transcript_file_path = os.path.join(merged_path, f"merged_{index}_TRANSCRIPT.csv")
            hog_feature = extract_clnf_hog_features(hog_file_path, transcript_file_path)

            hog_feature_path = f"../Embedding/merged_{index}_TRANSCRIPT_embedding/video_hog_emb.npy"
            np.save(hog_feature_path, hog_feature)

            audio_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/audio_emb.npy"))
            question_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/ellie_emb.npy"))
            answer_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/participant_emb.npy"))
            video_shape = np.shape(hog_feature)

            try:
                assert audio_shape[0] == question_shape[0] == answer_shape[0] == video_shape[0]
            except AssertionError:
                print("participant index", index)
                print("audio_shape", audio_shape)
                print("question_shape", question_shape)
                print("answer_shape", answer_shape)
                print("video_shape", video_shape)
                pass

# get_clnf_hog_features()

def get_clnf_other_features():
    def read_clnf_other_txt(folder_path):
        base_folder_path = os.path.basename(folder_path) # remove directory
        index = int(base_folder_path[:3])

        files = [
            f'{folder_path}/{index}_CLNF_features.txt',
            f'{folder_path}/{index}_CLNF_features3D.txt',
            f'{folder_path}/{index}_CLNF_AUs.txt',
            f'{folder_path}/{index}_CLNF_gaze.txt',
            f'{folder_path}/{index}_CLNF_pose.txt'
        ]

        # store each feature set in a dictionary with its corresponding filename
        feature_sets = {}

        for filename in files:
            # read the data into a pandas DataFrame, ", " is the separator
            df = pd.read_csv(filename, sep=", ")

            # extract columns after "success"
            success_index = df.columns.get_loc("success")
            features_after_success = df.iloc[:, success_index + 1:]

            # extract the feature set name from the filename
            # feature_set_name = filename.split('.')[1].split('_')[1]

            basename = os.path.basename(filename)  # remove directory
            feature_set_name = '_'.join(basename.split('.')[0].split('_')[1:3])  # g

            # assert the type of "features_after_success.values" is float64
            assert features_after_success.values.dtype == np.float64

            # save the features to our dictionary, using the feature set name as the key
            feature_sets[feature_set_name] = features_after_success.values

        # concatenate all features in the feature_sets dict
        all_features = np.concatenate(list(feature_sets.values()), axis=1)

        return all_features

    def extract_clnf_other_features(folder_path, transcript_file):
        clnf_data = read_clnf_other_txt(folder_path)


        time_values = np.arange(0, len(clnf_data) / 30, 1 / 30)

        df = pd.read_csv(transcript_file)
        # df = df[df['speaker'] == 'Participant'] # Filter out only the Participant's utterances

        utterance_features = []
        for i in range(len(df) - 1): # Ensure that only extract QA pairs
            if df.iloc[i]['speaker'] == 'Ellie' and df.iloc[i + 1]['speaker'] == 'Participant':
                q = df.iloc[i]['value']
                a = df.iloc[i + 1]['value']
                # pair_num = i // 2 + 1

                # Segment the text with Spacy
                q_doc = nlp(q)
                a_doc = nlp(a)

                # Check the word count
                if len(q_doc) >= 3 or len(a_doc) >= 3:
                    start_time, stop_time = df.iloc[i + 1]['start_time'], df.iloc[i + 1]['stop_time']

                    # Add a 1/30 second buffer to the start and stop times
                    start_time = start_time - 1/30
                    stop_time = stop_time + 1/30

                    indices = np.where((time_values >= start_time) & (time_values <= stop_time))[0]

                    try:
                        utterance_clnf = np.mean(clnf_data[indices], axis=0)
                    except TypeError:
                        debug = clnf_data[indices]
                        pass
                    utterance_clnf = np.nan_to_num(utterance_clnf)

                    utterance_features.append(utterance_clnf)
        return np.array(utterance_features)


    dataset_path = r"F:\数据集收集\抑郁检测-DAIC-WOZ\189SAMPLES"
    merged_path = r"D:\[SYNC]代码\Coding\[2023.06.13]Prompt-DAICWOZ\Corpus\Merged Transcript"

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
        # if os.path.isdir(folder_path) and "367" in folder_path:
            index = int(folder[:3])
            print(f"Processing {folder} CLNF feature...")

            transcript_file_path = os.path.join(merged_path, f"merged_{index}_TRANSCRIPT.csv")

            clnf_feature = extract_clnf_other_features(folder_path, transcript_file_path)

            clnf_feature_path = f"../Embedding/merged_{index}_TRANSCRIPT_embedding/video_clnf_emb.npy"
            np.save(clnf_feature_path, clnf_feature)

            audio_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/audio_emb.npy"))
            question_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/ellie_emb.npy"))
            answer_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/participant_emb.npy"))
            video_shape = np.shape(clnf_feature)

            try:
                assert audio_shape[0] == question_shape[0] == answer_shape[0] == video_shape[0]
            except AssertionError:
                print("participant index", index)
                print("audio_shape", audio_shape)
                print("question_shape", question_shape)
                print("answer_shape", answer_shape)
                print("video_shape", video_shape)
                pass

# get_clnf_other_features()

def get_participant_labels():
    import pandas as pd
    import os

    # Load csv files into pandas dataframes
    train_df = pd.read_csv('./train_split_Depression_AVEC2017.csv')
    dev_df = pd.read_csv('./dev_split_Depression_AVEC2017.csv')

    # Concatenate the two dataframes
    df = pd.concat([train_df, dev_df])

    # Set the participant ID as the index for easy access
    df.set_index('Participant_ID', inplace=True)

    # Get a list of all directories in the Embedding folder
    directories = os.listdir('../Embedding')

    for dir in directories:
        if dir.startswith('merged_') and dir.endswith('_TRANSCRIPT_embedding'):
            participant_id = int(dir.split('_')[1])  # Extract participant ID from the directory name
            if participant_id in df.index:  # Check if this ID is in our dataframe
                scores = df.loc[participant_id]  # Get the scores for this participant

                # Construct the string to write to the file
                try:
                    scores_str = '\n'.join([f'{name}\t{int(value)}' for name, value in scores.items()])
                except ValueError:
                    pass

                # Write the scores to the label.txt file in the corresponding directory
                with open(f'../Embedding/{dir}/label.txt', 'w') as file:
                    file.write(scores_str)

# get_participant_labels()

def get_multi_type_text_embedding(emb_type=None):
    import os
    import numpy as np
    import pandas as pd
    import spacy
    import re
    # import gensim

    if emb_type is None:
        raise ValueError("Please specify the type of embedding to extract.")

    elif emb_type == "glove":
        with open('./Glove_Preprocess/word2id.txt', 'r', encoding='utf-8') as f:
            word2id = eval(f.read())

        glove_embedder = np.load('./Glove_Preprocess/glove_embedding.npy')

    elif emb_type == "sbert":
        import torch
        from torch.nn import functional as F
        from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
        from sentence_transformers import SentenceTransformer

        sbert_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    elif emb_type == "fasttext":
        import fasttext.util
        fasttext.util.download_model('en', if_exists='ignore')  # English
        fasttext_embedder = fasttext.load_model('cc.en.300.bin')

    # function to calculate average vector of a sentence
    def get_sentence_vector(sentence):

        feature = None
        if emb_type is None:
            raise ValueError("Please specify the type of embedding to extract.")

        elif emb_type == "glove":
            tokens = sentence.replace('.', '').strip().split()

            embs = []
            for token in tokens:
                emb = glove_embedder[word2id[token.lower()]]
                embs.append(emb)
            if len(embs) == 0: # avoid empty list
                feature = np.array([0.] * 300)
            else:
                feature = np.mean(np.array(embs), 0, keepdims=False)

        elif emb_type == "sbert":
            feature = sbert_embedder.encode(sentence, convert_to_tensor=True)
            feature = F.normalize(feature, p=2, dim=-1)
            feature = feature.detach().cpu().numpy()

        elif emb_type == "fasttext":
            sentence = sentence.replace('.', '').strip()
            if len(sentence) == 0: # avoid empty sentence
                sentence = '<unknown>'

            feature = fasttext_embedder.get_sentence_vector(sentence)

        assert feature is not None

        return feature



    text_dir = f"{corpus_path}/Merged Transcript"
    for file in os.listdir(text_dir):
        if file.endswith('.csv') and "merged" in file:
        # if file.endswith('.csv') and "merged" in file and file[7:10] in ["381"]:
            print(f"Processing: {file}")

            # folder_name = file[:-4]
            # if not os.path.exists(folder_name):
            #     os.makedirs(folder_name)

            df = pd.read_csv(os.path.join(text_dir, file))
            questions = []
            answers = []


            for i in range(len(df) - 1):
                if df.iloc[i]['speaker'] == 'Ellie' and df.iloc[i + 1]['speaker'] == 'Participant':
                    q = df.iloc[i]['value']
                    a = df.iloc[i + 1]['value']
                    # pair_num = i // 2 + 1

                    # Segment the text with Spacy
                    q_doc = nlp(q)
                    a_doc = nlp(a)

                    # Check the word count
                    if len(q_doc) >= 3 or len(a_doc) >= 3:
                    # if len(q.split()) >= 3 or len(a.split()) >= 3:

                        # with open(f'{folder_name}/question_{pair_num}.txt', 'w') as f:
                        #     f.write(q)
                        # with open(f'{folder_name}/answer_{pair_num}.txt', 'w') as f:
                        #     f.write(a)

                        if '(' in q and ')' in q:  # 遇到Ellie缩写的情况，直接用括号内的自然语言问题
                            # print(question)
                            # print(patient_id)
                            q = re.findall(r"[(](.*?)[)]", q)[0]
                            # print(q)

                        questions.append(q)
                        answers.append(a)



            q_vecs = [get_sentence_vector(q) for q in questions]
            a_vecs = [get_sentence_vector(a) for a in answers]

            embedding_dir = f'../Embedding/{os.path.splitext(file)[0]}_embedding'

            np.save(os.path.join(embedding_dir, f'question_{emb_type}_emb.npy'), np.array(q_vecs))

            np.save(os.path.join(embedding_dir, f'text_{emb_type}_emb.npy'), np.array(a_vecs))

        # with open(f'{embedding_dir}/questions.txt', 'w') as f:
            #     f.write('\n'.join(questions))
            # with open(f'{embedding_dir}/answers.txt', 'w') as f:
            #     f.write('\n'.join(answers))

# get_multi_type_text_embedding('glove')
# get_multi_type_text_embedding('sbert')
# get_multi_type_text_embedding('fasttext')

def get_covarep_embedding():

    def read_covarep_csv(folder_path):
        base_folder_path = os.path.basename(folder_path) # remove directory
        index = int(base_folder_path[:3])

        files = [
            f'{folder_path}/{index}_COVAREP.csv',
            f'{folder_path}/{index}_FORMANT.csv',
        ]

        # store each feature set in a dictionary with its corresponding filename
        feature_sets = {}

        for filename in files:
            df = pd.read_csv(filename, header=None)
            features = df.values

            basename = os.path.basename(filename)  # remove directory
            feature_set_name = basename.split('.')[0].split('_')[1]  # feature name

            # assert the type of "features_after_success.values" is float64
            assert features.dtype == np.float64

            # save the features to our dictionary, using the feature set name as the key
            feature_sets[feature_set_name] = features

        # get the minimum length of all features
        min_length = min([len(feature) for feature in feature_sets.values()])

        # concatenate all features with same min_length in the feature_sets dict
        for feature_name in feature_sets.keys():
            feature_sets[feature_name] = feature_sets[feature_name][:min_length]
            assert len(feature_sets[feature_name]) > 10000

        all_features = np.concatenate(list(feature_sets.values()), axis=1)

        return all_features

    def extract_covarep_features(folder_path, transcript_file):
        covarep_data = read_covarep_csv(folder_path)

        time_values = np.arange(0, len(covarep_data) / 100, 1 / 100) # sampled every 0.01s

        df = pd.read_csv(transcript_file)
        # df = df[df['speaker'] == 'Participant'] # Filter out only the Participant's utterances

        utterance_features = []
        for i in range(len(df) - 1): # Ensure that only extract QA pairs
            if df.iloc[i]['speaker'] == 'Ellie' and df.iloc[i + 1]['speaker'] == 'Participant':
                q = df.iloc[i]['value']
                a = df.iloc[i + 1]['value']
                # pair_num = i // 2 + 1

                # Segment the text with Spacy
                q_doc = nlp(q)
                a_doc = nlp(a)

                # Check the word count
                if len(q_doc) >= 3 or len(a_doc) >= 3:
                    start_time, stop_time = df.iloc[i + 1]['start_time'], df.iloc[i + 1]['stop_time']

                    # Add a 1/100 second buffer to the start and stop times
                    start_time = start_time - 1/100
                    stop_time = stop_time + 1/100

                    indices = np.where((time_values >= start_time) & (time_values <= stop_time))[0]

                    try:
                        utterance_covarep = np.mean(covarep_data[indices], axis=0)
                    except TypeError:
                        debug = covarep_data[indices]
                        pass
                    utterance_covarep = np.nan_to_num(utterance_covarep)

                    utterance_features.append(utterance_covarep)
        return np.array(utterance_features)


    dataset_path = r"F:\数据集收集\抑郁检测-DAIC-WOZ\189SAMPLES"
    merged_path = r"D:\[SYNC]代码\Coding\[2023.06.13]Prompt-DAICWOZ\Corpus\Merged Transcript"

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
        # if os.path.isdir(folder_path) and "300" in folder_path:
            index = int(folder[:3])
            print(f"Processing {folder} COVAREP feature...")

            transcript_file_path = os.path.join(merged_path, f"merged_{index}_TRANSCRIPT.csv")

            covarep_feature = extract_covarep_features(folder_path, transcript_file_path)

            covarep_feature_path = f"../Embedding/merged_{index}_TRANSCRIPT_embedding/audio_covarep_emb.npy"
            np.save(covarep_feature_path, covarep_feature)

            question_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/ellie_emb.npy"))
            answer_shape = np.shape(np.load(f"../Embedding/merged_{index}_TRANSCRIPT_embedding/participant_emb.npy"))
            audio_shape = np.shape(covarep_feature)

            try:
                assert audio_shape[0] == question_shape[0] == answer_shape[0]
            except AssertionError:
                print("participant index", index)
                print("audio_shape", audio_shape)
                print("question_shape", question_shape)
                print("answer_shape", answer_shape)
                pass

# get_covarep_embedding()

def get_augmentation_embedding(emb_type=None):
    import os
    import numpy as np
    import pandas as pd
    import spacy
    import re
    # import gensim

    if emb_type is None:
        raise ValueError("Please specify the type of embedding to extract.")

    elif emb_type == "glove":
        with open('./Glove_Preprocess/word2id.txt', 'r', encoding='utf-8') as f:
            word2id = eval(f.read())

        glove_embedder = np.load('./Glove_Preprocess/glove_embedding.npy')

    elif emb_type == "sbert":
        import torch
        from torch.nn import functional as F
        from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
        from sentence_transformers import SentenceTransformer

        sbert_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    elif emb_type == "fasttext":
        import fasttext.util
        fasttext.util.download_model('en', if_exists='ignore')  # English
        fasttext_embedder = fasttext.load_model('cc.en.300.bin')

    # function to calculate average vector of a sentence
    def get_sentence_vector(sentence):

        feature = None
        if emb_type is None:
            raise ValueError("Please specify the type of embedding to extract.")

        elif emb_type == "glove":
            tokens = sentence.replace('.', '').strip().split()

            embs = []
            for token in tokens:
                emb = glove_embedder[word2id[token.lower()]]
                embs.append(emb)
            if len(embs) == 0: # avoid empty list
                feature = np.array([0.] * 300)
            else:
                feature = np.mean(np.array(embs), 0, keepdims=False)

        elif emb_type == "sbert":
            feature = sbert_embedder.encode(sentence, convert_to_tensor=True)
            feature = F.normalize(feature, p=2, dim=-1)
            feature = feature.detach().cpu().numpy()

        elif emb_type == "fasttext":
            sentence = sentence.replace('.', '').strip()
            if len(sentence) == 0: # avoid empty sentence
                sentence = '<unknown>'

            feature = fasttext_embedder.get_sentence_vector(sentence)

        assert feature is not None

        return feature



    participant_dir = f"../Embedding"
    for folder_name in os.listdir(participant_dir):
        folder_path = os.path.join(participant_dir, folder_name)

        synthetic_path = os.path.join(folder_path, 'syn_answers.txt')

        if os.path.exists(synthetic_path):
            print(f"Processing: {synthetic_path}")

            synthetic_embs = []

            with open(synthetic_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for line_idx, line in enumerate(lines):
                    synthetic_emb = []
                    results = line.strip().split('[SEP]')
                    try:
                        assert len(results) == 3
                    except AssertionError:
                        print(f"Error in synthetic data {synthetic_path}, line{line_idx+1}: {line}")

                    for result in results:
                        synthetic_emb.append(get_sentence_vector(result))
                    synthetic_embs.append(synthetic_emb)

            synthetic_embs = np.array(synthetic_embs) # (qa_num, 3, 300)
            np.save(os.path.join(folder_path, f'synthetic_{emb_type}_emb.npy'), synthetic_embs)

        # with open(f'{embedding_dir}/questions.txt', 'w') as f:
            #     f.write('\n'.join(questions))
            # with open(f'{embedding_dir}/answers.txt', 'w') as f:
            #     f.write('\n'.join(answers))

get_augmentation_embedding('glove')
