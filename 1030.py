#Mstplotlibの描写
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO

record1=next(SeqIO.parse("NC_045512_Wuhan.fa","fasta"))
record2=next(SeqIO.parse("sars.fa","fasta"))

seq1=record1.seq
seq2=record2.seq

def dotmatrix(f1,f2,win):
    record1=next(SeqIO.parse(f1,"fasta"))
    record2=next(SeqIO.parse(f2,"fasta"))

st.title("Dot matrix")

file1=st.sidebar.file_uploader("Sequence file 1:")
file2=st.sidebar.file_uploader("Sequence file 2:")

win=st.sidebar.slider("Window size:",4,100,10)

from io import StringIO

if file1 and file2:
    with StringIO(file1.getvalue().decode("utf-8")) as f1,\
        StringIO(file2.getvalue().decode("utf-8"))  as f2:
        dotmatrix(f1,f2,win)
         
win=10
len1=len(seq1)-win+1
len2=len(seq2)-win+1

width=500
height=500

image=np.zeros((height,width))

hash={}

for x in range(len1):
    sub1=seq1[x:win+x]
    if sub1 not in hash:
        hash[sub1]=[]
    hash[sub1].append(x)

for y in range(len2):
    sub2=seq2[y:win+y]
    py=int(y/len2*height)
    if sub2 in hash:
        for x in hash[sub2]:
            px=int(x/len1*width)
            image[py,px]=1

plt.imshow(image,extent=(1,len1,len2,1),cmap="Grays")

#plt.show()

st.pyplot(plt)