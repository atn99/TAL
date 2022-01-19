import nltk 
import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
#from nltk import RegexpParser
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger')

file_in = sys.argv[1]
file_out = sys.argv[2]

print("file_in :" , file_in, "& file_out :", file_out)

f1 = open(file_in, "r")
f2 = open(file_out, "w") 
for line in f1 :
    line_pos_tag = pos_tag((word_tokenize(line)))
    for word in line_pos_tag : 
        f2.write('\t'.join(word))
        f2.write('\n')
    


