import argparse
import pickle
import requests
import xmltodict
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os


def extract_url_text(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


def text2_docs_metadatas(filter,sitemap):
    xml = requests.get(sitemap).text
    raw_data = xmltodict.parse(xml)
    pages = []
    for info in raw_data['urlset']['url']:
        old_url = info['loc']
        last_slash_index = old_url.rfind('/')
        url = old_url[:last_slash_index] + '/go-internals' + old_url[last_slash_index:]  
        if filter in url:
            print(f"Processing URL: {url}")
            pages.append({'text': extract_url_text(url), 'source': url})

    text_splitter = CharacterTextSplitter(chunk_size=2000, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")
    return docs,metadatas

def save_pkl(docs,metadatas,model_path):
    store = FAISS.from_texts(docs, LlamaCppEmbeddings(model_path=model_path), metadatas=metadatas)
    print(docs)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

if __name__ == '__main__':
    filter_url = 'http://books.studygolang.com/go-internals'
    sitemap_url = 'http://books.studygolang.com/go-internals/sitemap.xml'
    model_path = "/data/cpp/origin-ggml-q8_0.bin"
    docs = text2_docs_metadatas(filter_url,sitemap_url)[0]
    metadatas = text2_docs_metadatas(filter_url,sitemap_url)[1]
    save_pkl(docs,metadatas,model_path)
    