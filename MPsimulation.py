import cProfile
import profile
import time


import matplotlib.pyplot as plt
import numpy as np

from Configuration import Configuration
from MPMotorUnitPool import MotorUnitPool
from InterneuronPool import InterneuronPool
from NeuralTract import NeuralTract
from SynapsesFactory import SynapsesFactory

from multiprocessing import Process, Array
SIZE_A, SIZE_B = 10000, 80000

def initialize():
    conf = Configuration('confTest.rmto')

    pools = dict()
    pools[0] = MotorUnitPool(conf, 'SOL')
    #pools[1] = NeuralTract(conf, 'CMExt')
    #pools[2] = InterneuronPool(conf, 'RC', 'ext')

    Syn = SynapsesFactory(conf, pools)

    t = np.arange(0.0, conf.simDuration_ms, conf.timeStep_ms)

    return pools, t

def simulator(pools, t, rank, nbrProcesses):

    processChunk = len(pools[0].unit) / (nbrProcesses)
    processUnits = range(rank * processChunk, rank * processChunk + processChunk)
    print processChunk
    print processUnits

    somaV = np.zeros_like(t)

    for i in xrange(0, len(t)):
        for j in xrange(len(pools[0].unit)):
            pools[0].iInjected[0] = 10
        #pools[1].atualizePool(t[i])
        pools[0].atualizeMotorUnitPool(t[i], processUnits)
        #pools[2].atualizeInterneuronPool(t[i])
        somaV[i] = pools[0].v_mV[1]

    pools[0].listSpikes()
    
    #plt.figure()
    #plt.plot(t, pools[0].Muscle.force, '-')

    #plt.figure()
    #plt.plot(t, dendV, '-')

    print pools[0].G.size

    plt.figure()
    plt.plot(t, somaV, '-')
    plt.figure()
    plt.plot(pools[0].poolSomaSpikes[:, 0],
             pools[0].poolSomaSpikes[:, 1]+1, '.')

    
if __name__ == '__main__':

    tic = time.time()
    pools, t = initialize()

    # TODO Optimize this size
    sharedItems = SIZE_A * SIZE_B
    sharedArray = Array('i', sharedItems, lock=False)
    # TODO no need for numpy arrays
    sharednpArray = np.frombuffer(sharedArray, dtype=int)
    sharednpArray = sharednpArray.reshape(SIZE_A, (SIZE_B-40000))

    nbrProcesses = 2
    for i in xrange(nbrProcesses):
        p = Process(target=simulator, args=(pools, t, i, nbrProcesses))
        p.start()
    print "==="
    print "Main process awaits with join"
    print "==="
    p.join()

    plt.show()
    toc =time.time()
    print str(toc - tic) + ' seconds'
