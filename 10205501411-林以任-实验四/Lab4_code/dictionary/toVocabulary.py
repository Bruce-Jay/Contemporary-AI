import csv

with open('2000kanji.txt', 'r', encoding='utf-8') as f:
    text = f.read()
char_list = list(text)

with open('vocabulary.csv', 'w', encoding='utf-8') as f:
    # for i, char in enumerate(char_list):
    #     f.write(str(i+1) + '\t' + char + '\n')
    writer = csv.writer(f)
    for i in range(1, len(char_list)):
        writer.writerow([i, char_list[i]])
