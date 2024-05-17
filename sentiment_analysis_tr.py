import numpy as np
import re #regular expressions(düzenli ifadeler), metindeki bazı karakterleri temizlemek için
import nltk 
import nltk as nlp #nltk(nlp), metin işleme kütüphanesidir
import pandas as pd #pandas: Veri manipülasyonu ve analizi için kullanılır (veri kümelerini yüklemek ve işlemek için kullanılır)
from nltk.corpus import stopwords  #metindeki durdurma(gereksiz) kelimeleri temizlemek için kullanırız

#bazı nlp dosyalarını indirme
nltk.download('stopwords')   
stopWords = set(stopwords.words('turkish'))
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#eğitim veri seti
df_train = pd.read_csv("C:/Users/Ahmet/Desktop/T3AI'LE/TEKNİK KOMİSYON/NLP/Sentimated Analiz/train.csv",encoding='unicode_escape')
df_train.head()


#test veri seti
df_test = pd.read_csv("C:/Users/Ahmet/Desktop/T3AI'LE/TEKNİK KOMİSYON/NLP/Sentimated Analiz/test.csv",encoding='unicode_escape')
df_test.head()

def pre_processing(text):
    #tüm harfleri küçük yap
    text = text.lower()  
    #noktalam işaretlerinden kurtulmak için...
    text = re.sub("^['abcçdefgğhıijklmnoöprsştuüvyz']"," ",text)#NOT: ^ çok önemli bunların dışında anlamı veriyor #noktalam işaretlerinden kurtulmak için... alfabedeki harfler ve boşluk dışındaki her karakteri temzizle
    
    #tokenize edioruz her birini ayırma 
    text = nlp.word_tokenize(text)  

    text = [word for word in text if not word in set(stopwords.words('turkish'))] # cümleye etki etmeyen kelimelerden kurtulma örn: "ya"

    lemma = nlp.WordNetLemmatizer() #lemmataizon işlemi: kelimenin ekini kökünü bulmaya çalışıyoruz
    text = [lemma.lemmatize(word) for word in text] #kelime kökünü bulur
    
    #cümleyi kurma
    text = ' '.join(text) #kelimeler arasına boşluk bırak ve birleşitir
    
    return text

#ön işleme(pre processing) sokulan yorumlar clean_text sütununa atıyor
df_test["clean_text"] = df_test["comment"].apply(lambda x: pre_processing(x))
df_train["clean_text"] = df_train["comment"].apply(lambda x: pre_processing(x))

print(df_test)
print("--------------------------------------------------------------------------")
print(df_train)

X_train = df_train["clean_text"]
X_test = df_test["clean_text"]
y_train = df_train["Label"]
y_test = df_test["Label"]

# print("x_train",X_train.shape)
# print("x_test",X_test.shape)
# print("y_train",y_train.shape)
# print("y_test",y_test.shape)

# #MODEL EĞİTİMİ 
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression  #sınıflandırma işlemi için
from sklearn.pipeline import Pipeline  
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer: vektörlerştirme işlemi için 
from sklearn.metrics import confusion_matrix

LogisticRegression = Pipeline(  [ ('tfidf', TfidfVectorizer()), ('clf', LogisticRegression()) ]  )
LogisticRegression.fit(X_train,y_train)

def plot_confusion_matrix(Y_test, Y_preds): #confusion matrixini çıkarıyor #pred: tahmin
    conf_mat = confusion_matrix(Y_test, Y_preds)
    #print(conf_mat)
    fig = plt.figure(figsize=(6,6))
    plt.matshow(conf_mat,cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(2), range(2))
    plt.xticks(range(2), range(2))
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(i-0.2, j+0.1, str(conf_mat[j,i]), color='tab:red')

print('\n')

#Skorlar
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

cv_scores = cross_val_score(LogisticRegression, X_train, y_train, cv=10) #x_train ve y_traini 10 parçaya bölüyor ve 10'unun ortlamasını alıyor
print('CV average score: %.2f' % cv_scores.mean())

print("------------------------------------")

result = LogisticRegression.predict(X_test) #x_testeki verileri tahmin ettirip resultlara alıyoruz
cr = classification_report(y_test, result) #gerçek değerler(y_test) ile resultları değerlemdiröek için metin raporu oluşturur
print(cr)

y_pred = LogisticRegression.predict(X_test)
   #sırasıyla precision, precision, f1 skorları
print(precision_score(y_test, y_pred, average='macro'), ': is the precision score')
print(recall_score(y_test, y_pred, average='macro'), ': is the recall score' )
print(f1_score(y_test, y_pred, average='macro'), ': is the fprecision1 score')

#COMMENT ÇEKME(SCRAPING)    #eğer bir sayfadaki cümleleri çekip analiz etmek isterseniz yoruma alınmış bu bölümü düzenleyip web scraping yapabilirsiniz

# import requests
# from bs4 import BeautifulSoup

# comments = []

# url = 'your url'

# response =requests.get(url)     #doğrulamalar için verify=False(opsiyonel)

# soup  = BeautifulSoup(response.content, 'html.parser')

#          ...

etext = input("Enter your text: ")
comment_list=[]
comment_list.append(etext)
for i in range(len(comment_list)):

  prediction=LogisticRegression.predict([comment_list[i]])
  proportion=LogisticRegression.predict_proba([comment_list[i]])

  if prediction[0]==1:
    print(comment_list[i]," is: ",proportion[0][1]," (Positive)")
  else:
    print(comment_list[i]," is: ",proportion[0][0]," (Negative)")
