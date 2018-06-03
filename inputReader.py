
# reads input provided as
# dataTrain.txt labelTrain.txt
# dataTest.txt

def readX(f, limit):
    matlist = []
    for line in f:
        if limit == 0:
            break
        linelist = []
        for word in line.split():
            if word == 'NaN':
                value = None
            else:
                value = float(word)
            linelist.append(value)
        matlist.append(linelist)
        limit = limit - 1
    return matlist

def readY(f, limit):
    matlist = []
    for line in f:
        if limit == 0:
            break
        linelist = []
        for word in line.split():
            value = int(word)
            linelist.append(value)
        matlist.append(linelist)
        limit = limit - 1
    return matlist

def read_nm(f):
    nl = f.readline().split()
    return int(nl[0]), int(nl[1])

def readTrainInput(predLimit = -1):
    f = open("dataTrain.txt", "r")
    n, m = read_nm(f) 
    if predLimit == -1:
        predLimit = n
    resultX = readX(f, predLimit)
    f = open("labelTrain.txt", "r")
    n, m = read_nm(f)
    resultY = readY(f, predLimit)
    return resultX, resultY

def readTestInput(predLimit = -1):
    f = open("dataTest.txt", "r")
    n, m = read_nm(f) 
    if predLimit == -1:
        predLimit = n
    resultX = readX(f, predLimit)
    return resultX

if __name__ == '__main__':
    pass