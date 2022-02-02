def confusion_matrix_(y, y_hat, labels=None):
    conc = np.array([y,y_hat]).T
    if labels:
        new = []
        for row in conc:
            
            true = 1
            for cel in row:
                if(cel not in labels):
                    true = 0
            if true == 1:
                new += [row]
        conc = np.array(new)
    key = np.unique(conc)
    dicoVal = {}
    for index, val in enumerate(key):
        dicoVal[val] = index
    res = np.zeros((len(key),len(key)))
    for row in conc:
        res[dicoVal[row[0]]][dicoVal[row[1]]] += 1
    return res

def countTest(y, y_hat, true = 1):
    tp, fp, tn, fn = 0,0,0,0
    for valY, valHat in zip(y,y_hat):
        if valY == valHat:
            tp += 1
        else:
            fp += 1
    return tp, fp, tn, fn

def precision_score_(y, y_hat, pos_label=1):
    tp, fp, tn, fn = countTest(y, y_hat, true = pos_label)
    return tp / (tp + fp)
