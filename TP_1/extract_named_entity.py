import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import RegexpParser

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('words')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('state_union')


file_in = sys.argv[1] #wsj_0010_sample.txt
file_out = sys.argv[2] #wsj_0010_sample.txt.ne.nltk

print("file_in :" , file_in, "& file_out :", file_out)

f1 = open(file_in, "r")
f2 = open(file_out, "w")


#tokenized = custom_sent_tokenizer.tokenize(sample_text)



for line in f1:

    words = nltk.word_tokenize(line)
    tagged = nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged)



    for subtree in namedEnt.subtrees() :
        if subtree.label() == 'ORGANIZATION' or subtree.label() =='PERSON' or\
                subtree.label() == 'LOCATION' or subtree.label() == 'DATE' \
                or subtree.label() == 'TIME' or subtree.label() == 'MONEY' \
                or subtree.label() == 'PERCENT' or subtree.label() == 'FACILITY'\
                or subtree.label() == 'GPE':
            f2.write(str(subtree))
            f2.write('\n')






"""
sample_text = state_union.raw("wsj_0010_sample.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))
"""


