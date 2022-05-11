import augmentations.rad as rad
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    folder = './mujoco_samples/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
