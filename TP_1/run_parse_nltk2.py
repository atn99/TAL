import nltk 
import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import RegexpParser

nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger')

file_in = sys.argv[1] #wsj_0010_sample.txt
file_out = sys.argv[2] #wsj_0010_sample.txt.chk.nltk

print("file_in :" , file_in, "& file_out :", file_out)

f1 = open(file_in, "r")
f2 = open(file_out, "w") 
grammar = """Compound:{<JJ>?<JJ>?<NN><NN>?}"""
chunker = RegexpParser(grammar)
print("After Regex:",chunker)

for line in f1 : 
    line_pos_tag = pos_tag((word_tokenize(line)))
    output = chunker.parse(line_pos_tag)
    
    for subtree in output.subtrees() : 
        if subtree.label() == 'Compound':
            f2.write(str(subtree))
            f2.write('\n')
