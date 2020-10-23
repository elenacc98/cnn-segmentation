import os
import timeit

ik = 0
def Cycle_Volume_Crop():
    global ik

    start = timeit.default_timer()
    for ik in range(5):
        os.chdir(mainInputDataDirectoryNAS)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Crop()

    stop = timeit.default_timer()
    print('Time: ', stop - start)