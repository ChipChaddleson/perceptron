
def broke(s, l):
    b = []
    for i in range(len(s)):
        b.append(int(s[i]))
    while len(b) < l:
        b.insert(0, 0)
    return(b)

l = 11

trainingInputs = [broke(str(bin(i).removeprefix("0b")), l) for i in range(10)]
labels = [1 if item[-3] == 1 else 0 for item in trainingInputs]

testData = [broke(str(bin(i).removeprefix("0b")), l) for i in range(10, 20)]
testLabels = [1 if item[-3] == 1 else 0 for item in testData]

print(testLabels)