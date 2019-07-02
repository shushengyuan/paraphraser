import collections
import os
import re
import torch as t
from torch.autograd import Variable
import numpy as np
import pandas as pd

# 清洗数据
def clean_str(string):
    '''
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    '''
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'\W+', ' ', string)
    string = string.lower()
    return string.strip()


class BatchLoader:
    def __init__(self, vocab_size=20000, sentences=None, datasets={'quora'}, path=''):
        '''
            Build vocab for sentences or for data files in path if None. 
        '''
        # vocabulary
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_vec = {}
        self.max_seq_len = 0

        # label
        self.unk_label = '<unk>'
        self.end_label = '</s>'
        self.go_label = '<s>'

        # data
        self.df_from_file = None
        self.sampling_file_name = None
        self.datasets = datasets
        self.quora_data_files = [path + 'data/quora/train.csv', path + 'data/quora/test.csv']
        self.snli_path = '../InferSent/dataset/SNLI/'
        self.mscoco_path = path + 'data/mscoco/'
        self.glove_path = '/home/aleksey.zotov/InferSent/dataset/GloVe/glove.840B.300d.txt'

        # read data from dataset
        if sentences is None:
            self.read_train_test_dataset()

        # build vocabulary
        self.build_vocab(sentences)
    
    ''' 以下是input block: 1 '''
    '''
        number：0
        输出：encoder_input_source, encoder_input_target
    '''
    def get_encoder_input(self, sentences):
        return [Variable(t.from_numpy(
            self.embed_batch([s + [self.end_label] for s in q]))).float() for q in sentences]
            # 结尾加上 end_label
    
    '''
        number：1
        输出：decoder_input_source, decoder_input_target
    '''
    def get_decoder_input(self, sentences): 
        enc_inp = self.embed_batch([s + [self.end_label] for s in sentences[0]]) 
        # input from encoder
        dec_inp = self.embed_batch([[self.go_label] + s for s in sentences[1]]) 
        # input from decoder ?
        return [Variable(t.from_numpy(enc_inp)).float(), Variable(t.from_numpy(dec_inp)).float()]
        # return two decoder's inputs
        
    '''
        number: 2
        output: the all inputs;
                target;  
    '''
    def input_from_sentences(self, sentences):
        sentences = [[clean_str(s).split() for s in q] for q in sentences]
        # clean
        encoder_input_source, encoder_input_target = self.get_encoder_input(sentences)
        decoder_input_source, decoder_input_target = self.get_decoder_input(sentences)
        target = self.get_target(sentences)

        return [encoder_input_source, encoder_input_target, 
                decoder_input_source, decoder_input_target,
                target]
    
    '''
        quote:  2
        number: 2.1
        output: target_idx
    '''
    def get_target(self, sentences):
        sentences = sentences[1]    # sentences[1] means target
        max_seq_len = np.max([len(s) for s in sentences]) + 1   # sequence's length
        target_idx = [[self.get_idx_by_word(w) for w in s]  
                        + [self.get_idx_by_word(self.end_label)] * (max_seq_len - len(s))
                        for s in sentences] 
        # target_onehot = self.get_onehot_wocab(target_idx)
        return Variable(t.from_numpy(np.array(target_idx, dtype=np.int64))).long()

    ''' endblock:   1 '''
    
    def get_raw_input_from_sentences(self, sentences):
        sentences = [clean_str(s).split() for s in sentences] 
        return Variable(t.from_numpy(self.embed_batch(sentences))).float()

    ''' block:  2 '''
    
    def next_batch(self, batch_size, type, return_sentences=False, balanced=True):
        if type == 'train':
            file_id = 0
        if type == 'test':
            file_id = 1
            
        # introduce dataset
        if balanced:
            df = pd.DataFrame()
            length = batch_size//len(self.datasets)
            if 'quora' in self.datasets:
                df = df.append(self.quora[file_id].sample(length, replace=False), ignore_index=True)
            if 'snli' in self.datasets:
                df = df.append(self.snli[file_id].sample(length, replace=False), ignore_index=True)
            if 'mscoco' in self.datasets:
                df = df.append(self.mscoco[file_id].sample(length, replace=False), ignore_index=True)
        else:
            df = self.data[file_id].sample(batch_size , replace=False)

        sentences = [df['question1'].values, df['question2'].values]
        
        # swap source and target
        # why random ?
        if np.random.rand() < 0.5: 
            sentences = [sentences[1], sentences[0]]
            # sentences[0] ：source
            # sentences[1] ：target 
        # quoto:    2    
        input = self.input_from_sentences(sentences)

        if return_sentences:
            return input, [[clean_str(s).split() for s in q] for q in sentences]    # ?
        else:
            return input

        
    def next_batch_from_file(self, batch_size, file_name, return_sentences=False):
        if self.sampling_file_name is None \
            or self.sampling_file_name != file_name \
            or self.df_from_file is None:

            self.sampling_file_name = file_name
            self.cur_file_point = 0

            predefined_datasets = {
                'snli_test': self.get_nli , 
                'quora_test': self.get_quora, 
                'mscoco_test': self.get_mscoco,
                'snips': self.get_snips
            }

            if file_name in predefined_datasets.keys():
                self.df_from_file = predefined_datasets[file_name]()[1]
            else:
                self.df_from_file = pd.read_csv(file_name)

            self.df_from_file = self.df_from_file.sample(
                min(self.df_from_file.shape[0], 6000), replace=False)

            print('{} sentences loaded from {}.'.format(self.df_from_file.shape[0], file_name))
            sentences = list(self.df_from_file['question1']) + list(self.df_from_file['question2'])
            # ADD new words to emb dict
            self.build_input_vocab(sentences)
        
        # file ends
        if self.cur_file_point == len(self.df_from_file):
            self.cur_file_point = 0
            return None

        end_point = min(self.cur_file_point + batch_size, len(self.df_from_file))
        df = self.df_from_file.iloc[self.cur_file_point:end_point]
        sentences = [df['question1'].values, df['question2'].values]
        self.cur_file_point = end_point

        input = self.input_from_sentences(sentences)

        if return_sentences:
            return input, [[clean_str(s).split() for s in q] for q in sentences]
        else:
            return input
        
    ''' endblock:   2 '''
    
    # Original taken from https://github.com/facebookresearch/InferSent/blob/master/data.py
    def embed_batch(self, batch):
        max_len = np.max([len(x) for x in batch])
        embed = np.zeros((len(batch), max_len, 300))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] == self.go_label or batch[i][j] == self.end_label:
                    continue
                if batch[i][j] in self.word_vec.keys(): 
                    embed[i, j, :] = self.word_vec[batch[i][j]]
                else:
                    embed[i, j, :] = self.word_vec['null']

        return embed

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent.split():
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<p>'] = ''
        word_dict['null'] = ''
        return word_dict
    
    def build_most_common_vocab(self, sentences):
        word_counts = collections.Counter(sentences)
        self.idx_to_word = [x[0] for x in word_counts.most_common(self.vocab_size - 2)] \
        + [self.unk_label] + ['</s>']
        self.word_to_idx = {self.idx_to_word[i] : i for i in range(self.vocab_size)}
    
    def sample_word_from_distribution(self, distribution):
        assert distribution.shape[-1] == self.vocab_size
        ix = np.random.choice(range(self.vocab_size), p=distribution.ravel())
        return self.idx_to_word[ix]

    '''找到最像的词'''
    def likely_word_from_distribution(self, distribution):
        assert distribution.shape[-1] == self.vocab_size
        ix = np.argmax(distribution.ravel())
        # argmax返回的是最大数的索引
        # 将多维数组降位为一维
        return self.idx_to_word[ix]
    
    def get_onehot_vocab(self, ids):
        batch_size = len(ids)
        max_seq_len = np.max([len(x) for x in ids])
        res = np.zeros((batch_size, max_seq_len, self.vocab_size), dtype=np.int32)
        for i in range(batch_size):
            for j in range(max_seq_len):
                if j < len(ids[i]):
                    res[i][j][ids[i][j]] = 1 
                else :
                    res[i][j][self.vocab_size - 1] = 1 # end symb
        return res

    def get_word_by_idx(self, idx):
        return self.idx_to_word[idx]

    def get_idx_by_word(self, w):
        if w in self.word_to_idx.keys():
            return self.word_to_idx[w]
        return self.word_to_idx[self.unk_label]

    def build_glove(self, word_dict):
        # create word_vec with glove vectors
        
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    self.word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(self.word_vec), len(word_dict)))

    def get_sentences_from_data(self):
        sentences = []
        for df in self.data:
            sentences += list(df['question1'].values) + list(df['question2'].values)
        return sentences

    def build_input_vocab(self, sentences):
        word_dict = self.get_word_dict(sentences)
        self.build_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    def build_output_vocab(self, sentences):
        self.max_seq_len = np.max([len(s) for s in sentences]) + 1
        text = ' '.join(sentences).split()
        self.build_most_common_vocab(text)

    def build_vocab(self, sentences):
        if sentences is None:
            sentences = self.get_sentences_from_data()
        sentences = [clean_str(s) for s in sentences]

        self.build_input_vocab(sentences)
        self.build_output_vocab(sentences)
        
        
    # READ DATA 
    def read_train_test_dataset(self):
        self.data = [pd.DataFrame(), pd.DataFrame()]

        if 'quora' in self.datasets:
            self.quora = self.get_quora()
            print('QUORA: train: {}, test: {}'.format(len(self.quora[0]), len(self.quora[1])))
            self.data = [d.append(q, ignore_index=True) for d,q in zip(self.data, self.quora)]

        if 'snli' in self.datasets:    
            self.snli = self.get_nli()
            print('SNLI: train: {}, test: {}'.format(len(self.snli[0]), len(self.snli[1])))
            self.data = [d.append(s, ignore_index=True) for d,s in zip(self.data, self.snli)]

        if 'mscoco' in self.datasets:    
            self.mscoco = self.get_mscoco()
            print('MSCOCO: train: {}, test: {}'.format(len(self.mscoco[0]), len(self.mscoco[1])))
            self.data = [d.append(m, ignore_index=True) for d,m in zip(self.data, self.mscoco)]
        
        print('ALL: train: {}, test: {}'.format(len(self.data[0]), len(self.data[1])))
    
    def get_snips(self): 
        intents_list = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 
                'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
        intent_to_id = dict([x[::-1] for x in enumerate(intents_list)])

        data_path = '../snips-intent-recognition/data/nlu-benchmark/' + \
            '2017-06-custom-intent-engines/intent_data/intent_data_train.csv'
        snips_df = pd.read_csv(data_path)

        X = snips_df['request']
        y = np.array([intent_to_id[x] for x in snips_df.drop(['request'],
                             axis=1).idxmax(axis=1)], dtype=np.int32)
        pairs = [[],[]]
        for intent_id in range(len(intents_list)):
            X_intent = X[y == intent_id]
            for i in range(len(X_intent) - 1):
                # RANDOM!
                j = np.random.choice(list(range(i+1, len(X_intent))))
                pairs[0].append(X_intent.iloc[i])
                pairs[1].append(X_intent.iloc[j])

        result_df = pd.DataFrame(data=np.array(pairs).T, columns=['question1', 'question2'])
        return [], result_df

    def get_quora(self):
        return [pd.read_csv(f)[['question1', 'question2']] for f in self.quora_data_files]

    def get_mscoco(self):
        return [pd.read_csv(self.mscoco_path + 'train.csv'),
                pd.read_csv(self.mscoco_path + 'valid.csv')]

    def get_nli(self):
        # https://github.com/facebookresearch/InferSent (c)
        data_path = self.snli_path
        
        s1 = {}
        s2 = {}
        target = {}

        dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

        for data_type in ['train', 'dev', 'test']:
            s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
            s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
            s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
            target[data_type]['path'] = os.path.join(data_path,
                                                     'labels.' + data_type)

            s1[data_type]['sent'] = np.array([line.rstrip() for line in
                                     open(s1[data_type]['path'], 'r')])
            s2[data_type]['sent'] = np.array([line.rstrip() for line in
                                     open(s2[data_type]['path'], 'r')])
            target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                    for line in open(target[data_type]['path'], 'r')])

            assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
                len(target[data_type]['data'])


        train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
                 'label': target['train']['data']}

        tr1 = s1['train']['sent'][target['train']['data'] == 0] # entailment
        tr2 = s2['train']['sent'][target['train']['data'] == 0]
        train_df = pd.DataFrame(data=np.array([tr1, tr2]).T, columns=['question1', 'question2'])

        ts1 = s1['test']['sent'][target['test']['data'] == 0] # entailment
        ts2 = s2['test']['sent'][target['test']['data'] == 0]
        test_df = pd.DataFrame(data=np.array([ts1, ts2]).T, columns=['question1', 'question2'])
        return [train_df, test_df]
            
