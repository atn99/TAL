import sys

file_in = sys.argv[1] #wsj_0010_sample.pos.ref
file_out = sys.argv[2] #wsj_0010_sample.pos.stanford.ref

print("file_in :" , file_in, "& file_out :", file_out)

f1 = open(file_in, "r")
f2 = open(file_out, "w")


for line in f1:
        if(not line.isspace()):
            #case if there is something writen in the line
            line_split = line.split("\t")
            word = line_split[0]
            pos_tag = line_split[1]
            pos_tag = pos_tag.replace("]","") # removing the ']' caracter
            pos_tag_without_space = pos_tag.replace(" ", "")
            pos_tag_without_space_and_return_line = pos_tag_without_space.split('\n')[0]
            if(pos_tag_without_space_and_return_line == '.'):
                f2.write(word + "_" + pos_tag_without_space_and_return_line)
            else:
                f2.write(word + "_" + pos_tag_without_space_and_return_line + " ")

        else:
            f2.write('\n')
            # case if this is an empty line


