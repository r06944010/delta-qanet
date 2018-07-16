from tqdm import tqdm
import ujson as json
import subprocess
from scipy.spatial import distance
import os
import csv

_num = 1500

opt1 = []
opt2 = []
opt3 = []
opt4 = []
indexing = []

home = os.path.expanduser("~")
file = os.path.join("kgb", "0622.json")

with open(file, "r") as fh:
    source = json.load(fh)
    for _id in range(len(source)):
        options = source[_id]['options']
        opt1.append(options[0].replace("''", '" ').replace("``", '" '))
        opt2.append(options[1].replace("''", '" ').replace("``", '" '))
        opt3.append(options[2].replace("''", '" ').replace("``", '" '))
        opt4.append(options[3].replace("''", '" ').replace("``", '" '))
        indexing.append(source[_id]["id"])

with open('kgb/opt1.csv', 'w') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(_num):  
        s.writerow([opt1[i].replace('\n','')])
with open('kgb/opt2.csv', 'w') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(_num):  
        s.writerow([opt2[i].replace('\n','')])
with open('kgb/opt3.csv', 'w') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(_num):  
        s.writerow([opt3[i].replace('\n','')])
with open('kgb/opt4.csv', 'w') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(_num):  
        s.writerow([opt4[i].replace('\n','')])

os.system('cat kgb/kgb_answer.csv | ./../../fastText-0.1.0/fasttext print-sentence-vectors \
						  /home/geneping/corpus/advdl/wiki_zh_model.bin > kgb/ans_emb.txt')
os.system('cat kgb/opt1.csv | ./../../fastText-0.1.0/fasttext print-sentence-vectors \
						  /home/geneping/corpus/advdl/wiki_zh_model.bin > kgb/opt1_emb.txt')
os.system('cat kgb/opt2.csv | ./../../fastText-0.1.0/fasttext print-sentence-vectors \
						  /home/geneping/corpus/advdl/wiki_zh_model.bin > kgb/opt2_emb.txt')
os.system('cat kgb/opt3.csv | ./../../fastText-0.1.0/fasttext print-sentence-vectors \
						  /home/geneping/corpus/advdl/wiki_zh_model.bin > kgb/opt3_emb.txt')
os.system('cat kgb/opt4.csv | ./../../fastText-0.1.0/fasttext print-sentence-vectors \
						  /home/geneping/corpus/advdl/wiki_zh_model.bin > kgb/opt4_emb.txt')

f =  open('kgb/ans_emb.txt','r')
ans = f.readlines()
f =  open('kgb/opt1_emb.txt','r')
opt1 = f.readlines()
f =  open('kgb/opt2_emb.txt','r')
opt2 = f.readlines()
f =  open('kgb/opt3_emb.txt','r')
opt3 = f.readlines()
f =  open('kgb/opt4_emb.txt','r')
opt4 = f.readlines()


f = open('kgb/final_ans.csv', 'w')
s = csv.writer(f,delimiter=',',lineterminator='\n')
s.writerow(['ID','Answer'])
for i in range(_num):
	a = ans[i].split(' ')[-301:-1]
	q = opt1[i].split(' ')[-301:-1]
	w = opt2[i].split(' ')[-301:-1]
	e = opt3[i].split(' ')[-301:-1]
	r = opt4[i].split(' ')[-301:-1]
	a = list(map(float, a))
	q = list(map(float, q))
	w = list(map(float, w))
	e = list(map(float, e))
	r = list(map(float, r))
	qq = distance.cosine(a, q)
	ww = distance.cosine(a, w)
	ee = distance.cosine(a, e)
	rr = distance.cosine(a, r)
	values = [qq,ww,ee,rr]
	choice = values.index(min(values)) + 1
	s.writerow([indexing[i], choice])
f.close()