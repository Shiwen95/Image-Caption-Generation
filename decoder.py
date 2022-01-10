"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from vocabulary import Vocabulary
from utils import *
from config import *
import string
from sklearn.metrics.pairwise import cosine_similarity

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)
print(len(vocab))


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=0, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here


    # for each batch, prepare the targets using this torch.nn.utils.rnn function
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    for epoch in range(NUM_EPOCHS):
        for i, (feature, caption, lengths) in enumerate(train_loader):
            target = pack_padded_sequence(caption, lengths, batch_first=True)[0]
            outputs = decoder.forward(feature, caption, lengths)
            loss = criterion(outputs, target)
            if (i % 100 == 0):
                print(loss, i)
            loss.backward()
            optimizer.step()
            decoder.zero_grad()

    # save model after training
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")



# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])

    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)

    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm



#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################


    # TODO define decode_caption() function in utils.py
    # predicted_caption = decode_caption(word_ids, vocab)

    n=0
    bleu_score=0
    references_5 = []
    dict_cs = {}
    dict_bl = {}
    dict_cap = {}
    dict_cs_rescale = {}

    for test_image_id, test_cleaned_caption in zip(test_image_ids, test_cleaned_captions):
        img = Image.open(IMAGE_DIR+test_image_id+'.jpg')
        img = data_transform(img).unsqueeze(0)
        feature = encoder.forward(img).squeeze().unsqueeze(0)
        sampled_ids = decoder.sample(feature)
        predicted_caption = decode_caption(sampled_ids,vocab)


#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report

        references_5.append(test_cleaned_caption)

        n+=1
        if n==5:
            print('===========================================')
            print('\nThe imageid is:{}'.format(test_image_id))
            print('\nThe references are:')
            for refer in references_5:
                print(refer)
            print('\nThe prediction caption is:')
            print(predicted_caption)

            dict_cap[test_image_id] = [references_5, predicted_caption]

            # BLEU for evaluation: BLEUEvaluation function define in utils.py
            bleu_score = BLEUEvaluation(references_5, predicted_caption)
            print('\nThe blueScore is:',bleu_score)

            # cosine similarity: COSINEEvaluation function define in utils.py
            cosine_sim = COSINEEvaluation(decoder,vocab,predicted_caption, references_5)
            print('\nThe cosine similarity is:',cosine_sim,'. After scaling, it is ',(cosine_sim + 1) / 2)

            dict_cs_rescale[test_image_id] = (cosine_sim + 1) / 2
            dict_cs[test_image_id] = cosine_sim
            dict_bl[test_image_id] = bleu_score

            references_5=[]
            bleu_score=0
            n=0

    LS_cs = list(dict_cs.values())
    LS_bl = list(dict_bl.values())
    LS_cs_rescale = list(dict_cs_rescale.values())
    print("The BLEU score's average value is {},\tmaximum value is {},\tminimum value is {}.".format(sum(LS_bl) / 1003,max(LS_bl), min(LS_bl)))
    print("The cosine score's average value is {},\tmaximum value is {},\tminimum value is {}.".format(sum(LS_cs) / 1003,max(LS_cs), min(LS_cs)))
    print("After rescaling, the cosine score's average value is {},\tmaximum value is {},\tminimum value is {}.".format(sum(LS_cs_rescale) / 1003,max(LS_cs_rescale),min(LS_cs_rescale)))







