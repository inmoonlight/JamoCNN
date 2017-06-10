# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import numpy as np
import json
import pickle
import sys
sys.path.append('../../')

from alarmoon import Alarmoon
import hangle


class Romanization():
    def __init__(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        
    def load_romanized(self, path_to_data):
        # path_to_data example: '../data/romanize_review.json'
        # preprocessing method should be differed from casy by case. 
        alphabet = self.alphabet
        examples = []
        labels = []
        with open(path_to_data) as f:
            i = 0
            # rate 10 --> 2, rate --> 1
            review = json.load(f)
            for r in review:
                stars = r["stars"]
                text = r["text"]
                text_end_extracted = self.extract_end(list(text.lower()))
                padded = self.pad_sentence(text_end_extracted)
                text_int8_repr = self.string_to_int8_conversion(padded, alphabet)
                if stars == 1:
                    labels.append([1, 0])
                    examples.append(text_int8_repr)
                else:
                    labels.append([0, 1])
                    examples.append(text_int8_repr)
                i += 1
                if i % 10000 == 0:
                    print("Instances processed: " + str(i))
        return examples, labels

    def extract_end(self, char_seq):
        if len(char_seq) > 1014:
            char_seq = char_seq[-1014:]
        return char_seq

    def pad_sentence(self, char_seq, padding_char=" "):
        char_seq_length = 1014
        num_padding = char_seq_length - len(char_seq)
        new_char_seq = char_seq + [padding_char] * num_padding
        return new_char_seq

    def string_to_int8_conversion(self, char_seq, alphabet):
        x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
        return x

    def get_batched_one_hot(self, char_seqs_indices, labels, start_index, end_index):
        alphabet = self.alphabet
        x_batch = char_seqs_indices[start_index:end_index]
        y_batch = labels[start_index:end_index]
        x_batch_one_hot = np.zeros(shape=[len(x_batch), len(x_batch[0]), len(alphabet), 1])
        for example_i, char_seq_indices in enumerate(x_batch):
            for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                if char_seq_char_ind != -1:
                    x_batch_one_hot[example_i][char_pos_in_seq][char_seq_char_ind][0] = 1
        return [x_batch_one_hot, y_batch]

    def load_data(self):
        examples, labels = self.load_romanized('../data/romanize_review.json')
        x = np.array(examples, dtype=np.int8)
        y = np.array(labels, dtype=np.int8)
        print("x_char_seq_ind=" + str(x.shape))
        print("y shape=" + str(y.shape))
        return [x, y]

    def batch_iter(self, x, y, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        # data = np.array(data)
        data_size = len(x)
        num_batches_per_epoch = int(data_size/batch_size) + 1
        for epoch in range(num_epochs):
            print("In epoch >> " + str(epoch + 1))
            print("num batches per epoch is: " + str(num_batches_per_epoch))
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x_shuffled = x[shuffle_indices]
                y_shuffled = y[shuffle_indices]
            else:
                x_shuffled = x
                y_shuffled = y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                x_batch, y_batch = self.get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
                batch = list(zip(x_batch, y_batch))
                yield batch

                
class Hangulization():
    def __init__(self):
        self.cnn_encoder = hangle.ConvolutionalNN_Encoder()
        self.jamo = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ "
        
    def load_jamo(self, path_to_data):
        # path_to_data example: '../data/balanced_raw_original.pkl'
        # preprocessing method should be differed from casy by case. 
        with open(path_to_data, 'rb') as f:
            data = pickle.load(f)
        jamo_reviews = []
        labels = []
        i = 0
        for rate, review in zip(data.rate, data.contents):
            # review
            review = review.replace('"', '')
            if review.startswith('관람객'):
                review = review[3:]
            strim_hangul = hangle.normalize(review, punctuation= False)
            rm_space_hangul = strim_hangul.replace(" ", '')
            jamos = []
            for hangul in rm_space_hangul:
                jamo = hangle.split_jamo(hangul)
                jamos.extend(jamo)
            jamo_reviews.append(jamos)
            # rate
            if rate == 1:
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            i += 1
            if i % 10000 == 0:
                print("Instances processed: " + str(i))
        return jamo_reviews, labels
    
    def pad_sentence(self, char_seq, padding_char=" "):
        char_seq_length = 420
        num_padding = char_seq_length - len(char_seq)
        char_seq.extend([padding_char] * num_padding)
        return char_seq

    def load_data(self):
        jamo_reviews, labels = self.load_jamo('../data/balanced_raw_original.pkl')
        # x is list 
        x = []
        for char_seq in jamo_reviews:
            new_char_seq = self.pad_sentence(char_seq)
            x.append(new_char_seq)
        # y is array
        y = np.array(labels)
        print("number of data: {}".format(len(x)))
        print("dimension of data: {}".format(len(x[0])))
        print("y shape: {}".format(y.shape))
        return [x, y]
    
    def get_batched_one_hot(self, x_shuffled, y_shuffled, start_index, end_index):
        cnn_encoder = self.cnn_encoder
        jamo = self.jamo
        x_batch = x_shuffled[start_index:end_index]
        y_batch = y_shuffled[start_index:end_index]
        x_batch_one_hot = np.zeros(shape=[len(x_batch), len(x_batch[0]), len(jamo), 1])
        for batch_i, jamo_seq in enumerate(x_batch):
            for jamo_pos, jamo_seq_jamo in enumerate(jamo_seq):
                jamo_ind = cnn_encoder.cvocabs[jamo_seq_jamo]
                if jamo_ind != -1:
                    x_batch_one_hot[batch_i][jamo_pos][jamo_ind][0] = 1
        return [x_batch_one_hot, y_batch]
    
    def get_batched_channel(self, x_shuffled, y_shuffled, start_index, end_index):
        cnn_encoder = self.cnn_encoder
        jamo = self.jamo
        x_batch = x_shuffled[start_index:end_index]
        y_batch = y_shuffled[start_index:end_index]
        x_batch_channel = np.zeros(shape=[len(x_batch), len(x_batch[0])//3, len(jamo), 3])
        for batch_i, jamo_seq in enumerate(x_batch):
            for jamo_pos, jamo_seq_jamo in enumerate(jamo_seq):
                jamo_ind = cnn_encoder.cvocabs[jamo_seq_jamo]
                if (jamo_ind != -1) & (jamo_pos % 3 == 0):
                    x_batch_channel[batch_i][jamo_pos//3][jamo_ind][0] = 1
                elif (jamo_ind != -1) & (jamo_pos % 3 == 1):
                    x_batch_channel[batch_i][jamo_pos//3][jamo_ind][1] = 1
                elif (jamo_ind != -1) & (jamo_pos % 3 == 2):
                    x_batch_channel[batch_i][jamo_pos//3][jamo_ind][2] = 1
        return [x_batch_channel, y_batch]
    
    def batch_iter(self, x, y, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(x)
        num_batches_per_epoch = int(data_size/batch_size) + 1
        for epoch in range(num_epochs):
            print("In epoch >> " + str(epoch + 1))
            print("num batches per epoch is: " + str(num_batches_per_epoch))
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x_shuffled = [x[shuffled_idx] for shuffled_idx in shuffle_indices]
                y_shuffled = y[shuffle_indices]
            else:
                x_shuffled = x
                y_shuffled = y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                x_batch, y_batch = self.get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
                batch = list(zip(x_batch, y_batch))
                yield batch
                
    def batch_iter_channel(self, x, y, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(x)
        num_batches_per_epoch = int(data_size/batch_size) + 1
        for epoch in range(num_epochs):
            print("In epoch >> " + str(epoch + 1))
            print("num batches per epoch is: " + str(num_batches_per_epoch))
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x_shuffled = [x[shuffled_idx] for shuffled_idx in shuffle_indices]
                y_shuffled = y[shuffle_indices]
            else:
                x_shuffled = x
                y_shuffled = y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                x_batch, y_batch = self.get_batched_channel(x_shuffled, y_shuffled, start_index, end_index)
                batch = list(zip(x_batch, y_batch))
                yield batch