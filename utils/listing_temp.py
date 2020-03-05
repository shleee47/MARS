import os


def listing(rootpath, fout):
    with open(fout, 'w') as outputfile:
        for root, subdirs, files in os.walk(rootpath):
            print(root, "//", subdirs, "//")
            print(len(files))





if __name__ == '__main__':
    listing('/home/sangbuem/MARS/dataset/Kinetics', 'a')
