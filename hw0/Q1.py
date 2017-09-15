import sys

f_path = sys.argv[1]

dic = {}

with open(f_path, 'r') as f:
	for line in f:
		for word in line.split():
			if word in dic :
				dic[word] = dic[word] + 1
			else : 
				dic[word] = 1
idx = 0
with open(f_path, 'r') as f, open('Q1.txt','w') as out_f:
	for line in f:
		for word in line.split():
			if word in dic :
				out_f.write(word+' '+str(idx)+' '+str(dic[word]))
				idx += 1
				del dic[word]
				if dic != {}:
					out_f.write('\n')
			else : 
				continue
