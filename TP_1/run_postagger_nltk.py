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
nb_lines = 0
for line in f1:
    line_pos_tag = pos_tag((word_tokenize(line)))
    if not line.isspace():
        for word in line_pos_tag :
            nb_lines += 1
            f2.write('\t'.join(word))
            if nb_lines != 110: #Particulier à notre cas, simplement pour ne pas mettre le retour chariot à la dernière ligne.
                f2.write('\n')
f1.close()
f2.close()

