from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from sklearn.metrics import precision_score, recall_score, accuracy_score
from numpy.random import multivariate_normal
import numpy as np


square = np.random.rand(1000, 2)
diameter = 0.4*0.4;
center = 0.5;

#randomly read in the coordinates
alldata = ClassificationDataSet(2, 1, nb_classes = 2)
for coordinate in square:
    if(coordinate[0] - center)**2 + (coordinate[1] - center)**2 <= diameter:
        klass = 0
    else:
        klass = 1
    alldata.addSample(coordinate, klass)
    
# split dataset into 75% training and 25% test set
tstdata, trndata = alldata.splitWithProportion( 0.25 )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

#outside of circle is 0
#inside of circle is 1
here, _ = where(trndata['class'] == 0)
there, _ = where(trndata['class'] == 1)

# Plotting the training data
plt.plot(trndata['input'][here, 0], trndata['input'][here, 1], 'bo')
plt.plot(trndata['input'][there, 0], trndata['input'][there, 1], 'ro')
plt.savefig('C:/Users/Kristen/Desktop/Train.png')

#building a network (6 layers)
fnn = buildNetwork(trndata.indim, 6, trndata.outdim, outclass = SoftmaxLayer)
#setting up a back propogation trainer
trainer = BackpropTrainer(fnn, dataset=trndata, momentum = 0.1, verbose = True, weightdecay = 0.01)

trainer.trainUntilConvergence(maxEpochs=15)

trnresult = percentError( trainer.testOnClassData(), 
                         trndata['class'] )
tstresult = percentError( trainer.testOnClassData( 
   dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs,     "  train error: %5.2f%%" % trnresult,     "  test error: %5.2f%%" % tstresult


out = fnn.activateOnDataset(tstdata)
out = out.argmax(axis=1)

# Plotting the test data
for point, test in enumerate(tstdata['class']):
    if test==out[point]:
        if test==0:
            plt.plot(tstdata['input'][point, 0], tstdata['input'][point, 1], 'bo') #plots the inside of the circle Blue
        else:
            plt.plot(tstdata['input'][point, 0], tstdata['input'][point, 1], 'ro') #plots the outside of the circle Red
    else:
        plt.plot(tstdata['input'][point, 0], tstdata['input'][point, 1], 'ko') #plots the errors Black

plt.savefig('C:/Users/Kristen/Desktop/Test.png')

# Prints the precision, accuracy, and recall
print "Precision:  ", precision_score(tstdata['class'], out)
print "Accuracy:  ", accuracy_score(tstdata['class'], out)
print "Recall:  ", recall_score(tstdata['class'], out)




