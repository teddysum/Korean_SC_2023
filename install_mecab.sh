# Install MeCab tokenizer (korean)
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xzf mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
ldconfig
cd ..

wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xzf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
autoreconf
./configure
make
make install
cd ..

git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996
python setup.py install
cd ..