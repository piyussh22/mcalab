from django.shortcuts import render
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# Create your views here.


def index(request):
    return render(request, 'mainapp/base.html')

def algo(request):
    # s="Apply the tokenization manually on the two sentences used in section 2 (“I’ve been waiting for a HuggingFace course my whole life.” and “I hate this so much!”). Pass them through the model and check that you get the same logits as in section 2. Now batch them together using the padding token, then create the proper"
    # target="the tokenization manually the two sentences used for a Hugging my whole"
    if request.method=='POST':
        s=request.POST['originalContent']
        target=request.POST['suspiciousContent']
        tar=target.split()
        t=s.split()
        lst1=[]
        lst2=[]
        for i in range(len(t)):
            lst1.append(' '.join(t[i:i+3]))
        for i in range(len(tar)):
            lst2.append(' '.join(tar[i:i+3]))
        count=0
        res=[]
        for i in range(len(lst2)):
            for j in range(len(lst1)):
                if lst2[i]==lst1[j]:
                    count=count+1
                    res.append(lst1[j])
            
        ans=[]
        print(len(res))
        ans.append(len(t))
        ans.append(len(tar))
        ans.append(count)
        ans.append(res)
        dict={'ans':ans,
        'data1':lst1,
        'data2':lst2
        }
        return render(request,'mainapp/algo.html',dict)

def pro(request):

    sentences = ["We are  the students of nitt and pursuing masters of computer applications",
             "other students of nitt are in masters",
             "we nittians are having computer applications",
             "this sentence is different"
    ]
    model = SentenceTransformer('nli-distilroberta-base-v2')
    sentence_embeddings = model.encode(sentences)
    cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    )
    similarity_score = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
    ).flatten() 
    # print(sentences[1:])
    ans=pd.DataFrame({"Sentence":sentences[1:],"Similarity_Score":similarity_score })
    dict={
       
        'ans':ans
    }
    # print(dict)
    return render(request,'mainapp/pro.html', dict)


    
