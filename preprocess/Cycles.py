import os
import timeit

ik = 0
def Cycle_Volume_Crop():
    global ik

    for ik in range(5):
        os.chdir(mainInputDataDirectoryNAS)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Crop()


def Cycle_Volume_Reshape():
    global ik

    for ik in range(5):
        os.chdir(mainInputDataDirectoryLoc)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Reshape()


def Cycle_Volume_Label():
    global ik

    for ik in range(5):
        os.chdir(mainInputDataDirectoryLoc)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Label()


def Cycle_Merge_Labels():
    global ik

    for ik in range(5):
        os.chdir(mainInputDataDirectoryLoc)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Merge_Labels()
