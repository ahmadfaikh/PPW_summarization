
import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Import library yang dibutuhkan untuk melakukan data preprocessing
st.title("Summarization")
import subprocess

@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
download_en_core_web_sm()
nlp = spacy.load('en_core_web_sm')
text = st.text_input("Input text")
# Ekstrak kalimat dari teks
# text = 'ini adalah text. ini juga test. aku juga text. ini sangat bersih juga saya. kemudian saya berbagi'
if text != '':
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Hitung TF-IDF dari kalimat
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=vectorizer.get_feature_names_out())
    # Hitung kesamaan kosinus antara kalimat
    cos_sim_result = []  # untuk menyimpan hasil cosine sim akhir
    graf_result = []  # untuk menyimpan hasil graf akhir
    treshold = 0.2  # inisialisasi treshold

    cos_sim_now = []
    graf_now = nx.DiGraph()
    cos_sim = cosine_similarity(tfidf_matrix)  # menjadikan tfidf ke cosine

    # cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Buat grafik dari kesamaan kosinus
    # inisialisasi indeks awal perulangan dari setiap hasil cosine
    for i_hasil in range(len(cos_sim)):
        arr_cosim = []

        # inisialisasi indeks kedua perulangan dari setiap hasil cosine
        for j_hasil in range(i_hasil+1, len(cos_sim)):
            # cek apakah cosim dari kalimat 1 dan 2 lebih dari treshold?
            if cos_sim[i_hasil][j_hasil] > treshold:
                # print(f'Similairty kalimat ke - {i_hasil} : {j_hasil} = {cos_sim[i_hasil][j_hasil]}')

                # menyimpan nilai indeks awal, indeks awal+1, hasil cosim
                arr_cosim.append([i_hasil, j_hasil, cos_sim[i_hasil][j_hasil]])
                # menyimpan nilai indeks awal, indeks awal+1, bobot=hasil cosim
                graf_now.add_edge(i_hasil, j_hasil,
                                  weight=cos_sim[i_hasil][j_hasil])

        cos_sim_now.append(arr_cosim)
        # graf_now.append(graf_current)
    cos_sim_result.append(cos_sim_now)
    graf_result.append(graf_now)
    closeness_centrality = nx.closeness_centrality(graf_result[0])

    # Hitung Closeness Centrality
    # closeness_centrality = nx.closeness_centrality(G)

    # Temukan kalimat paling penting berdasarkan Closeness Centrality
    most_important_sentence_idx = max(
        closeness_centrality, key=closeness_centrality.get)
    most_important_sentence = sentences[most_important_sentence_idx]
    my_dict = closeness_centrality
    imp_stc = ' '
    for i in range(3):
        get_stc = max(my_dict, key=my_dict.get)
        imp_stc = imp_stc+sentences[get_stc]
        my_dict.pop(get_stc)
    # print(imp_stc)
    st.write("Kalimat penting: ")
    st.write(imp_stc)
