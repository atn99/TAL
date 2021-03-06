#import nltk
import sys

mapping_table = sys.argv[1]  #(POSTags_PTB_Universal.txt)
#file we want to convert
file_in = sys.argv[2] #wsj_0010_sample.txt.pos.stanford & wsj_0010_sample.pos.stanford.ref
#file converted
file_out = sys.argv[3] #wsj_0010_sample.txt.pos.univ.stanford & wsj_0010_sample.txt.pos.univ.ref

#print("file_in :" , file_in, "& file_out :", file_out)

#dictonary for mapping
f1 = open(mapping_table, "r")
my_dict = {}
for line in f1:
    tuples = line.split()
    my_dict[tuples[0]] = tuples[1]


#extract file_in
f2 = open(file_in, "r")
f3 = open(file_out, "w")
for line in f2:
    word_pos_tag_list = line.split(" ")
    for word_pos_tag in word_pos_tag_list:
        pos_tag = word_pos_tag.split("_")

        if "\n" in pos_tag[1]:
            # the case if this is the end of the line
            pos_tag[1] = my_dict[pos_tag[1].replace("\n","")]
            pos_tag[1] = pos_tag[1] + "\n"
            f3.write(pos_tag[0] + "_" + pos_tag[1])
        else :
            pos_tag[1] = my_dict[pos_tag[1].replace("\n", "")]
            f3.write(pos_tag[0] + "_" + pos_tag[1] + " ")






    


