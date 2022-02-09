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

standards_label = {"ORGANIZATION": "ORG", "PERSON": "PERS", "LOCATION": "LOC", "DATE": "MISC", "TIME": "MISC",
                   "MONEY": "MISC", "PERCENT": "MISC", "FACILITY": "ORG", "GPE": "LOC"}

file_in = sys.argv[1]  # wsj_0010_sample.txt.ne.nltk
file_out = sys.argv[2]  # wsj_0010_sample.txt.ne.std.nltk

print("file_in :", file_in, "& file_out :", file_out)

f1 = open(file_in, "r")
f2 = open(file_out, "w")

for line in f1:
    line_split = line.split(" ")
    label_with_parenthese = line_split[0]
    label = label_with_parenthese[1:]

    standard_label = standards_label[label]

    f2.write(str(line_split[1:]) + standard_label)
    f2.write('\n')
