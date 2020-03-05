import os
import glob
import shutil


def del_dummydirs(rootpath, list):
    for root, subdirs, files in os.walk(rootpath):
        """
        walk through given rootpath, delete dirs in list
        """
        for s in subdirs:
            if s in list:
                shutil.rmtree(os.path.join(root, s))
                print("deleted - ", os.path.join(root, s))

        """
        walk through given rootpath, delete files in list
        """
        for f in files:
            if f in list:
                os.remove(os.path.join(root, f))
                print("deleted - ", os.path.join(root, f))

if __name__ == '__main__':
#    del_dummydirs('/home/nas/DB/DB_video-nonlocal-light/400_val', ['@eaDir', 'Thumbs.db'])
    del_dummydirs('/home/sangbuem/MARS/dataset/Kinetics', ['@e', 'Thumb'])
