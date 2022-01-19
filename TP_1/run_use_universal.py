import nltk 
import sys

mapping_table = sys.argv[1]
#file we want to convert
file_in = sys.argv[2] #wsj_0010_sample.txt.pos.nltk & wsj_0010_sample.pos.ref
#file converted
file_out = sys.argv[3] #wsj_0010_sample.txt.pos.univ.nltk & wsj_0010_sample.txt.pos.univ.ref

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
    tuples = line.split()   
    tuples[1] = my_dict[tuples[1]]
    f3.write('\t'.join(tuples))
    f3.write('\n')



    


