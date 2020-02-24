# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:41:18 2019

@author: Arifin
"""

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import ast
'''
jumData = 0

with open('130datalatih.txt') as file:
    pos, neg = [], []
    for element in file.readlines():
        infile = element.replace('\n','').split('\t')
        if infile[1] == 'Positif':
            pos.append(infile[0])
        else:
            neg.append(infile[0])
        jumData += 1
        

#stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
pos_stemmed = [stemmer.stem(str(doc)) for doc in pos]
neg_stemmed = [stemmer.stem(str(doc)) for doc in neg]

#print(pos_stemmed)
#print(neg_stemmed)

#tokenisasi
tokenize = lambda doc: doc.split(" ")
pos_token = [tokenize(doc) for doc in pos_stemmed]
neg_token = [tokenize(doc) for doc in neg_stemmed]

#print(pos_token)
#print(neg_token)

#filtering
stopwords=[]
with open("stopwords.txt") as a:
    content = a.readlines()
stopwords = [x.strip() for x in content]
filter = lambda doc: [w for w in doc if w not in stopwords]
pos_filtered = [filter(d) for d in pos_token]
neg_filtered = [filter(d) for d in neg_token]

#print(pos_filtered)
#print(neg_filtered)

#mendapatkan term unik
docFiltered = pos_filtered, neg_filtered

def term_unik(dokumen, terms):
    for docs in dokumen:
        for doc in docs:
            for term in doc:
                if term not in terms:
                    terms.append(term)
    return(terms)

terms = []
terms = term_unik(docFiltered, terms)
#print(terms)

#jumlah term
jum_term = len(terms)
#print(jum_term)
    
#raw tf
tf_pos = []
tf_neg = []
for term in terms:
    row = []
    for doc in pos_filtered:
        row.append(doc.count(term))
    tf_pos.append(row)

for term in terms:
    row = []
    for doc in neg_filtered:
        row.append(doc.count(term))
    tf_neg.append(row)

#print(tf_pos)
#print(tf_neg)

#jumlah term tiap kelas
term_sum_pos = {}
for i in range(0, len(terms)):
    term_sum_pos[terms[i]] = sum(tf_pos[i])

term_sum_neg = {}
for i in range(0, len(terms)):
    term_sum_neg[terms[i]] = sum(tf_neg[i])

#print(term_sum_pos)
#print(term_sum_neg)

#kemunculan term di semua dokumen
term_sum_alldoc = {}
for i in range(0, len(terms)):
    term_sum_alldoc[terms[i]] = sum(tf_pos[i] + tf_neg[i])
        
#print(term_sum_alldoc)

#pos neg
def count(tf_docs):
    x = []
    for doc in tf_docs:
        x.append(sum(doc))
    return sum(x)

count_pos = count(tf_pos)
count_neg = count(tf_neg)

#print(count_pos)
#print(count_neg)

#total kemunculan semua term
term_sum = 0
#term_sum = sum(term_sum_alldoc.values())
term_sum = count_pos + count_neg

#print(term_sum)

#prior
prior_pos = len(pos)/jumData
prior_neg = len(neg)/jumData

#print(prior_pos)
#print(prior_neg)

#likelihood
def likelihood(terms, count, term_sum):
    likelihood = {}
    for term in terms:
        likelihood[term] = (term_sum[term] + 1)/(count + len(terms))
    return likelihood

likelihood_pos = likelihood(terms, count_pos, term_sum_pos)
likelihood_neg = likelihood(terms, count_neg, term_sum_neg)

#evidence
def evidence(terms, termsumdoc, termsum):
    evidence = {}
    for term in terms:
        evidence[term] = termsumdoc[term]/termsum
    return evidence

evidence_allterms = evidence(terms, term_sum_alldoc, term_sum)
#print(evidence_allterms)


with open("termunik.txt", "w") as f:
    f.write(str(terms))

with open("likelihoodpos.txt", "w") as f:
    f.write(str(likelihood_pos))
    
with open("likelihoodneg.txt", "w") as f:
    f.write(str(likelihood_neg))
    
with open("evidence.txt", "w") as f:
    f.write(str(evidence_allterms))
    
with open("priorpos.txt", "w") as f:
    f.write(str(prior_pos))
    
with open("priorneg.txt", "w") as f:
    f.write(str(prior_neg))
'''
'=============================='
with open("termunik.txt", "r") as f:
    terms = ast.literal_eval(f.read())
    
with open("priorpos.txt", "r") as f:
    prior_pos = ast.literal_eval(f.read())
    
with open("priorneg.txt", "r") as f:
    prior_neg = ast.literal_eval(f.read())

with open("likelihoodpos.txt", "r") as f:
    likelihood_pos = ast.literal_eval(f.read())
    
with open("likelihoodneg.txt", "r") as f:
    likelihood_neg = ast.literal_eval(f.read())

with open("evidence.txt", "r") as f:
    evidence_allterms = ast.literal_eval(f.read())

with open('20datauji.txt') as test:
    datatest = []
    labels = []
    for element in test.readlines():
        data, label= element.replace('\n','').split("\t")
        datatest.append(data)
        labels.append(label)
        
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stem_test = [stemmer.stem(str(doc)) for doc in datatest]

tokenize = lambda doc: doc.split(" ")
token_test = [tokenize(dok) for dok in stem_test]

stopwords=[]
with open("stopwords.txt") as a:
    content = a.readlines()
stopwords = [x.strip() for x in content]
filter = lambda doc: [w for w in doc if w not in stopwords]
filter_test = [filter(d) for d in token_test]

#pembobotan kata berdasarkan lexicon
with open('barasa.txt', encoding="utf-8") as file:
    syn, lang, good, lem, pos, neg = [], [], [], [], [], []
    for element in file.readlines():
        infile = element.replace('\n','').split('\t')
        if infile[0]:
            syn.append(infile[0])
        if infile[1]:
            lang.append(infile[1])
        if infile[2]:
            good.append(infile[2])
        if infile[3]:
            lem.append(infile[3])
        if infile[4]:
            pos.append(infile[4])
        if infile[5]:
            neg.append(infile[5])

posscore = [float(i) for i in pos]
negscore = [float(i) for i in neg]

a = list(zip(lem , posscore, negscore))

dict_score = dict()

for i in a:
    if i[0] not in dict_score.keys():
        tmp = dict()
        tmp['pos'] = i[1]
        tmp['neg'] = i[2]
        dict_score[i[0]] = tmp
    else:
        dict_score[i[0]]['pos'] += i[1]
        dict_score[i[0]]['neg'] += i[2]
        
senti_score = dict()

for k,v in dict_score.items():
    total_score = v['pos'] + v['neg']
    senti_score[k] = dict()
    if total_score == 0:
        senti_score[k]['posScore'] = 0
        senti_score[k]['negScore'] = 0
    else:
        senti_score[k]['posScore'] = v['pos'] / total_score
        senti_score[k]['negScore'] = v['neg'] / total_score

#posterior dengan pembobotan lexicon
hasil = []
for docs in filter_test:
    like_pos = 1
    like_neg = 1
    evid_pos = 1
    evid_neg = 1
    for kata in docs:
        if kata in terms:# and kata in senti_score.keys():
            try:
                like_pos *= likelihood_pos[kata]+senti_score[kata]['posScore']
                like_neg *= likelihood_neg[kata]+senti_score[kata]['negScore']
                evid_pos *= evidence_allterms[kata]+senti_score[kata]['posScore']
                evid_neg *= evidence_allterms[kata]+senti_score[kata]['negScore']
                #print(kata)
            except KeyError:
                like_pos *= likelihood_pos[kata]
                like_neg *= likelihood_neg[kata]
                evid_pos *= evidence_allterms[kata]
                evid_neg *= evidence_allterms[kata]
                #print(kata,' no bobot')
    post_pos = (prior_pos * like_pos) / evid_pos
    post_neg = (prior_neg * like_neg) / evid_neg
    
    #print(post_pos)
    #print(post_neg)
    text = " ".join(docs)
    if post_pos > post_neg:
        hasil.append([text, "Positif"])
    else:
        hasil.append([text, "Negatif"])

print("Dengan Pembobotan Lexicon Based Features")
for i in range(len(hasil)):
    print('Data uji ke-',i+1,'dikasifikasikan ke dalam kelas: ',hasil[i][1])
    
print("===============================")
tp, fp, tn, fn = 0, 0, 0 ,0
akurasi, precission, recall, fmeasure = 0, 0, 0, 0

for i in range(len(hasil)):
    if hasil[i][1] == 'Positif' and labels[i] == 'Positif':
        tp += 1
    if hasil[i][1] == 'Negatif' and labels[i] == 'Negatif':
        tn += 1
    if hasil[i][1] == 'Positif' and labels[i] == 'Negatif':
        fp += 1
    if hasil[i][1] == 'Negatif' and labels[i] == 'Positif':
        fn += 1

akurasi = (tp + tn)/(tp + tn + fp + fn)
if tp == 0:
    precision = 0
    recall = 0
    fmeasure = 0
else:
    precission = tp/(tp + fp)
    recall = tp/(tp + fn)
    fmeasure = (2*precission*recall)/(precission + recall)
    
print("     Pengujian     ")
print("Akurasi =", akurasi)
print("Precission =", precission)
print("Recall =", recall)
print("F-Measure =", fmeasure)

print("===========================")

#posterior tanpa pembobotan lexicon
nb_hasil = []
for doc in filter_test:
    nb_like_pos = 1
    nb_like_neg = 1
    evid = 1
    for kata in doc:
        if kata in terms:
            nb_like_pos *= likelihood_pos[kata]
            nb_like_neg *= likelihood_neg[kata]
            evid *= evidence_allterms[kata]
            #print(kata)
    nb_post_pos = (prior_pos * nb_like_pos) / evid
    nb_post_neg = (prior_neg * nb_like_neg) / evid

    #print(nb_post_pos)
    #print(nb_post_neg)
    
    text = " ".join(doc)
    if nb_post_pos > nb_post_neg:
        nb_hasil.append([text, "Positif"])
    else:
        nb_hasil.append([text, "Negatif"])

print("Tanpa Pembobotan Lexicon Based Features")        
for i in range(len(nb_hasil)):
    print('Data uji ke-',i+1,'dikasifikasikan ke dalam kelas: ',nb_hasil[i][1])

nb_tp, nb_fp, nb_tn, nb_fn = 0, 0, 0 ,0
nb_akurasi, nb_precission, nb_recall, nb_fmeasure = 0, 0, 0, 0

for i in range(len(nb_hasil)):
    if nb_hasil[i][1] == 'Positif' and labels[i] == 'Positif':
        nb_tp += 1
    if nb_hasil[i][1] == 'Negatif' and labels[i] == 'Negatif':
        nb_tn += 1
    if nb_hasil[i][1] == 'Positif' and labels[i] == 'Negatif':
        nb_fp += 1
    if nb_hasil[i][1] == 'Negatif' and labels[i] == 'Positif':
        nb_fn += 1

nb_akurasi = (nb_tp + nb_tn)/(nb_tp + nb_tn + nb_fp + nb_fn)
if nb_tp == 0:
    nb_precision = 0
    nb_recall = 0
    nb_fmeasure = 0
else:
    nb_precission = nb_tp/(nb_tp + nb_fp)
    nb_recall = nb_tp/(nb_tp + nb_fn)
    fmeasure = (2*nb_precission*nb_recall)/(nb_precission + nb_recall)
    
print("==========Pengujian==========")
print("Akurasi =", nb_akurasi)
print("Precission =", nb_precission)
print("Recall =", nb_recall)
print("F-Measure =", nb_fmeasure)

"==========================="
with open('penilai1.txt') as p1:
    data1 = []
    kelas1 = []
    for element in p1.readlines():
        data, kelas= element.replace('\n','').split("\t")
        data1.append(data)
        kelas1.append(kelas)
        
with open('penilai2.txt') as p2:
    data2 = []
    kelas2 = []
    for element in p2.readlines():
        data, kelas= element.replace('\n','').split("\t")
        data2.append(data)
        kelas2.append(kelas)
        
with open('penilai3.txt') as p3:
    data3 = []
    kelas3 = []
    for element in p3.readlines():
        data, kelas= element.replace('\n','').split("\t")
        data3.append(data)
        kelas3.append(kelas)

pospos1, posneg1, negpos1, negneg1 = 0, 0, 0, 0
pospos2, posneg2, negpos2, negneg2 = 0, 0, 0, 0
pospos3, posneg3, negpos3, negneg3 = 0, 0, 0, 0


for i in range(len(kelas1)):
    if kelas1[i] == 'Positif' and kelas2[i] == 'Positif':
        pospos1 += 1
    if kelas1[i] == 'Positif' and kelas2[i] == 'Negatif':
        posneg1 += 1
    if kelas1[i] == 'Negatif' and kelas2[i] == 'Negatif':
        negneg1 += 1
    if kelas1[i] == 'Negatif' and kelas2[i] == 'Positif':
        negpos1 += 1
  
for i in range(len(kelas1)):
    if kelas1[i] == 'Positif' and kelas3[i] == 'Positif':
        pospos2 += 1
    if kelas1[i] == 'Positif' and kelas3[i] == 'Negatif':
        posneg2 += 1
    if kelas1[i] == 'Negatif' and kelas3[i] == 'Negatif':
        negneg2 += 1
    if kelas1[i] == 'Negatif' and kelas3[i] == 'Positif':
        negpos2 += 1
        
for i in range(len(kelas2)):
    if kelas2[i] == 'Positif' and kelas3[i] == 'Positif':
        pospos3 += 1
    if kelas2[i] == 'Positif' and kelas3[i] == 'Negatif':
        posneg3 += 1
    if kelas2[i] == 'Negatif' and kelas3[i] == 'Negatif':
        negneg3 += 1
    if kelas2[i] == 'Negatif' and kelas3[i] == 'Positif':
        negpos3 += 1

pa1, pe1, pa2, pe2, pa3, pe3 = 0, 0, 0, 0, 0, 0
    
pa1 = (pospos1 + negneg1)/(pospos1 + posneg1 + negpos1 + negneg1)
pe1 = ((pospos1 + posneg1) + (pospos1+negpos1) + (negpos1 + negneg1) + (posneg1 + negneg1))/((pospos1 + posneg1 + negpos1 + negneg1)*4)

pa2 = (pospos2 + negneg2)/(pospos2 + posneg2 + negpos2 + negneg2)
pe2 = ((pospos2 + posneg2) + (pospos2 + negpos2) + (negpos2 + negneg2) + (posneg2 + negneg2))/((pospos2 + posneg2 + negpos2 + negneg2)*4)

pa3 = (pospos3 + negneg3)/(pospos3 + posneg3 + negpos3 + negneg3)
pe3 = ((pospos3 + posneg3) + (pospos3 + negpos3) + (negpos3 + negneg3) + (posneg3 + negneg3))/((pospos3 + posneg3 + negpos3 + negneg3)*4)

kappa1, kappa2, kappa3 = 0, 0, 0

kappa1 = (pa1 - pe1)/(1 - pe1)
kappa2 = (pa2 - pe2)/(1 - pe2)
kappa3 = (pa3 - pe3)/(1 - pe3)
print("========================")
print("Kappa-Measure")
print("penilai 1 dengan penilai 2 :",kappa1)
print("penilai 1 dengan penilai 3 :",kappa2)
print("penilai 2 dengan penilai 3 :",kappa3)

