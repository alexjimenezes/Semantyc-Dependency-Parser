import math
f = open("dm.sdp", encoding="utf8")
f1 = open("train.sdp", "w")
f2 = open("test.sdp", "w")
n_sentences = 0
for row in f:
    if row[0] == "#":
        n_sentences += 1
print("There are ", n_sentences, "sentences")

train_sentences = math.modf(n_sentences * 0.8)[1]
test_sentences = n_sentences - train_sentences

f = open("dm.sdp", encoding="utf8")

i = 0
for row in f:
    if i < train_sentences:
        if i == train_sentences - 1 and row[0] == "#":
            f2.write(row)
        else:
            f1.write(row)
    else:
        f2.write(row)
    if row[0] == "#":
        i += 1
