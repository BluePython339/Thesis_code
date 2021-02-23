import os
import sys


def open_csv(fname):
    with open(fname, 'r') as f:
        data = f.readlines()

    csv_data = [i.strip('\n').split(',') for i in data]

    return csv_data

def get_file_size(file):
    return os.stat(file).st_size

def get_file_paths(csvdata, base_path ):
    fin = []
    apt = {}
    for i in csvdata:
        apt[i[1]] = 0
        path = base_path+i[0]
        fin.append((get_file_size(path), i[1]))
    return fin,apt

if __name__ == "__main__":
    a, apt = get_file_paths(open_csv("fixed.csv"), "/home/bluepython339/Documents/thesis/decompfiles/data/")
    aptfull = apt.copy()
    aptperc = apt.copy()
    b = sorted(a, key=lambda i : i[0])
    for i in b:
        aptfull[i[1]] += 1
        if 100 < i[0] > int(sys.argv[1]):
            apt[i[1]] += 1
    total_in_scope = 0
    total = 0
    for i in apt:
        total_in_scope += apt.get(i)
        aptperc[i] = (apt.get(i)/aptfull.get(i))*100

    for i in apt:
        print("group: {}, amount: {}".format(i,apt.get(i)))
    print('\n'*2)
    print("Total sample count: {}".format(total_in_scope))
    #exit()
    for i in aptfull:
        total += aptfull.get(i)
        print("group: {}, total amount: {} ".format(i, aptfull.get(i)))

    print("total set size: {}".format(total))
    exit()
    with open("train_set.csv", 'w+') as f:
        for i in range(1962):
            f.write("/content/drive/MyDrive/Colab Notebooks/datast/data/train/{}.json\n".format(i))

    with open("test_set.csv", 'w+') as f:
        for i in range(949):
            f.write("/content/drive/MyDrive/Colab Notebooks/datast/data/test/{}.json\n".format(i))







