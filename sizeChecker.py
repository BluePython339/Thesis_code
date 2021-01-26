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
        if i[0] > int(sys.argv[1]):
            apt[i[1]] += 1

    for i in apt:
        aptperc[i] = (apt.get(i)/aptfull.get(i))*100

    for i in aptperc:
        print("group: {}, percentage: {}%".format(i,aptperc.get(i)))





