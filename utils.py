import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *
import string
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []


    # QUESTION 1.1

    for line in lines:
        image_ids.append(line.split('.')[0])
        caption = line.split('\t')[1].strip()
        puncs=[' ,',' .'," '",'-',' "',' (',' )',' ;',' :',' &',' ?',' !','=']
        for p in puncs:
            caption=caption.replace(p,'')
        caption=caption.lower()
        cleaned_captions.append(caption)
    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words
    words={}
    wl=[]

    for caption in cleaned_captions:
        for word in caption.split(' '):
            words[word]=words.get(word,0)+1

    wordList=list(words.items())
    wordList.sort(key=lambda x:x[1],reverse=True)
    for w,n in wordList:
        if n<=3:
            break
        wl.append(w)

    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    for word in wl:
        vocab.add_word(word)

    return vocab



def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        ref_captions (str list): ground truth captions
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """

    predicted_caption = ""


    # QUESTION 2.1

    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    if '<start>' in sampled_caption:
        sampled_caption.remove('<start>')
    if '<end>' in sampled_caption:
        sampled_caption.remove('<end>')

    predicted_caption = ' '.join(sampled_caption)

    return predicted_caption



"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def BLEUEvaluation(references,predict):

    for i,reference in enumerate(references):
        references[i] = reference.split(" ")
    predict = predict.split(' ')
    score = sentence_bleu(references,predict,weights=(1,0,0,0))
    return  score

def COSINEEvaluation(decoder,vocab,predicted_caption,list_of_references):
    #the input of list_of_reference is a list of 5 lists with words as elements
    predicted_tensor = torch.zeros(1, len(predicted_caption.split(' ')))

    for i, word in enumerate(predicted_caption.split(' ')):
        predicted_tensor[0][i] = vocab.word2idx[word]

    predicted_vector = decoder.embed(predicted_tensor.long())
    predicted_vector = predicted_vector.squeeze(0)
    average_vector = predicted_vector.mean(0)

    sim = 0
    for r in list_of_references:

        ref_tensor = torch.zeros(1, len(r))
        for j, word in enumerate(r):
            ref_tensor[0][j] = vocab.__call__(word)

        ref_vector = decoder.embed(ref_tensor.long())
        ref_vector = ref_vector.squeeze(0)
        average_vector_ref = ref_vector.mean(0)

        sim += cosine_similarity(np.expand_dims(average_vector.detach().numpy(),0),
                                 np.expand_dims(average_vector_ref.detach().numpy(),0))

    return float(sim/len(list_of_references))