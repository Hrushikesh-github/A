import numpy as np

def rank5_accuracy(preds,labels):
    rank1=0
    rank5=0

    for (p,gt) in zip(preds,labels):
        # sort the probabilities by their index in descending order so that
        # the more confident guesses are at the front of the list

        p=np.argsort(p)[::-1]
        # check if the groud-truth label is in the top 5
        if gt in p[:5]:
            rank5+=1

        if gt==p[0]:
            rank1+=1

    rank1/=float(len(labels))
    rank5/=float(len(labels))
    
    # return a tuple of rank1 and rank5 accuracies
    return (rank1,rank5)

