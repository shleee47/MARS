import os
import shutil


def main(din):
    """
    move '.mpeg' file in given din into own directory
    """
    files = []
    for r, d, f in os.walk(din):    
        f.sort()
        for file in f:
            if '.mpeg' in file:
                file_dir = os.path.join(r, file.split(".")[0])
                if os.path.isfile(os.path.join(r, file)):
                    if not os.path.exists(file_dir):
                        os.mkdir(file_dir)
                    shutil.move(os.path.join(r, file), file_dir)

        break # prevent recursively visiting the sub-directories


if __name__ == '__main__':
    main('../dataset_video/ntu_cctv/fight/')
