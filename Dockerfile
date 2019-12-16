FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y python3-software-properties
RUN apt-get install -y python3 python3-dev python3-setuptools python3-pip
RUN pip3 install numpy scipy scikit-learn pandas fasttext gensim razdel catboost joblib sentencepiece pymorphy2 textdistance nltk
