import sentencepiece as spm
import csv


sp = spm.SentencePieceProcessor()
sp.load('../m.model')
encoded = sp.encode('五于天末开下')
decoded = sp.decode(encoded)
print(encoded)  # [98, 18, 94, 24, 28, 2, 13, 3, 61]
print(decoded)  # 五于天末开下