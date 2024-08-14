import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2024 
Programming Homework 1 - Trigram Language Models
Patarada Yontrarak
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    ngrams = []

    if n == 1:
        new_sequence = ['START'] + sequence + ['STOP']
    else:
        new_sequence = (n-1) * ['START'] + sequence + ['STOP']


    for j in range(0, len(new_sequence) - n + 1):

        # gets as many words as 'n'
        curr_words = new_sequence[j:j + n]

        # turns into tuple to follow format
        curr_words_tuple = tuple(curr_words)

        ngrams.append(curr_words_tuple)

    return ngrams




class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        self.num_sentence = 0
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)




    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)# might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)


        for i in corpus: # a sequence
            self.num_sentence += 1

            unigram_list = get_ngrams(i, 1)
            for j in unigram_list: # words in sequence
                self.unigramcounts[j] += 1

            bigram_list = get_ngrams(i, 2)
            for j in bigram_list:
                self.bigramcounts[j] += 1

            trigram_list = get_ngrams(i, 3)
            for j in trigram_list:
                self.trigramcounts[j] += 1

        self.totalcounts = sum(self.unigramcounts.values())
        self.totalcounts -= self.unigramcounts[('START',)]

        return



    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        bigram = (trigram[0], trigram[1])
        count_trigram = self.trigramcounts[trigram]
        count_bigram = self.bigramcounts[bigram]

        if bigram == ('START', 'START'):
            return count_trigram / self.num_sentence

        if count_bigram == 0:
            return 1/len(self.lexicon)

        return count_trigram / count_bigram



    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        unigram = (bigram[0],)
        count_unigram = self.unigramcounts[unigram]
        count_bigram = self.bigramcounts[bigram]

        if unigram == ('START',):
            return count_bigram / self.num_sentence

        if count_unigram == 0:
            return 1/len(self.lexicon)

        return count_bigram / count_unigram


    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        count_unigram = self.unigramcounts[unigram]
        probability_unigram = count_unigram / self.totalcounts

        return probability_unigram




    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        smooth_prob = (lambda1 * self.raw_trigram_probability(trigram) +
                       lambda2 * self.raw_bigram_probability(trigram[1:]) +
                       lambda3 * self.raw_unigram_probability(trigram[2:]))

        if smooth_prob == 0:
            return 1/self.totalcounts

        return smooth_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        log_prob = 0
        trigrams_list = get_ngrams(sentence, 3)

        for i in trigrams_list:
            prob = self.smoothed_trigram_probability(i)
            log_prob += math.log2(prob)
        return log_prob


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        M = 0
        sum_prob = 0

        # iterate to get sentences
        for i in corpus:
            sum_prob += self.sentence_logprob(i)
            M += len(i) + 1

        l = sum_prob / M

        perplexity = 2 ** (-l)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct +=1
            total += 1

    
        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp1 > pp2:
                correct += 1
            total += 1

        score = correct / total

        return score

if __name__ == "__main__":

    # print(get_ngrams(["natural","language","processing"],1))
    # print(get_ngrams(["natural","language","processing"],2))
    # print(get_ngrams(["natural","language","processing"],3))



    model = TrigramModel(sys.argv[1])
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])

    # print(model.sentence_logprob(["natural","language","processing"]))


    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt.


    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")
    print(acc)



