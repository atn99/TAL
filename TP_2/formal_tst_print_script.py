import sys

file_in = sys.argv[1] #formal-tst.NE.key.04oct95_small.txt.ne.stanford
#file converted
file_out = sys.argv[2] #formal-tst-inshape.NE.key.04oct95_small.txt.ne.stanford

#extract file_in
f1 = open(file_in, "r")
f2 = open(file_out, "w")
word_named_list = []
for line in f1:
    words = line.split(" ")
    for word in words:
        to_delete = "/O"
        if to_delete not in word:
            if not word.isspace():
                #f2.write(word + '\n')
                word_named_list.append(word)

#print(word_named_list)
number_of_words = len(word_named_list)
total_occurrence = 0
word_named_occurrence_dico = {}

for word_named in word_named_list:
    if word_named not in word_named_occurrence_dico.keys():
        occurrence = word_named_list.count(word_named)
        total_occurrence += occurrence
        word_named_occurrence_dico[word_named] = occurrence

#print(word_named_occurrence_dico)
assert number_of_words == total_occurrence

for element in word_named_occurrence_dico:
    occurrence = word_named_occurrence_dico.get(element)
    ratio = round(occurrence/number_of_words, 2)
    f2.write(element.replace('/', ' ') + " " + str(occurrence) + " " + str(ratio) +'\n')









