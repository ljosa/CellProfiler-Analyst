#!/usr/bin/env python

import sys
import csv
from optparse import OptionParser
import numpy as np
import cpa
from scipy.spatial.distance import cdist, cosine, euclidean, cityblock
# Import pyemd in a funny way so we only err if it's being used
try:
    from pyemd import emd
except ImportError:
    def emd(a, b):
        import pyemd
from .profiles import Profiles
from .confusion import confusion_matrix, write_confusion

def vote(predictions):
    votes = {}
    for i, prediction in enumerate(predictions):
        votes.setdefault(prediction, []).append(i)
    winner = sorted((len(indices), indices[0]) for k, indices in votes.items())[-1][1]
    return predictions[winner]

def crossvalidate(profiles, true_group_name, holdout_group_name=None):
    profiles.assert_not_isnan()

    true_labels = profiles.regroup(true_group_name)

    if holdout_group_name:
       holdouts = profiles.regroup(holdout_group_name)
    else:
       holdouts = None

    confusion = {}
    dist = cdist(profiles.data, profiles.data, 'cosine')
    keys = profiles.keys()
    for i, key in enumerate(keys):
       if key not in true_labels:
           continue
       true = true_labels[key]
       if holdouts:
          ho = tuple(holdouts[key])
          held_out = np.array([tuple(holdouts[k]) == ho for k in keys], dtype=bool)
          dist[i, held_out] = -1.
       else:
          dist[i, i] = -1.
       indices = np.argsort(dist[i, :])
       predictions = []
       for j in indices:
           if dist[i, j] == -1.:
               continue # Held out.
           if keys[j] not in true_labels:
               continue
           predictions.append(true_labels[keys[j]])
           if len(predictions) == 1:
               predicted = vote(predictions)
               confusion[true, predicted] = confusion.get((true, predicted), 0) + 1
               break
    return confusion


class NNClassifier(object):
    def __init__(self, features, labels, distance='cosine'):
        assert isinstance(labels, list)
        assert len(labels) == features.shape[0]
        self.features = features
        self.labels = labels
        self.distance = distance

    def classify(self, feature):
        all_zero = np.all(self.features == 0, 1)
        distances = np.array([self.distance(f, feature) if not z else np.inf
                              for f, z in zip(self.features, all_zero)])
        return self.labels[np.argmin(distances)]

# A second implementation, originally written to make it possible to
# incorporate SVA (now removed, but kept in the sva branch), but kept
# for now because it may be clearer than the implementation above.
def crossvalidate(profiles, true_group_name, holdout_group_name=None, 
                  train=NNClassifier, distance=cosine):
    profiles.assert_not_isnan()
    keys = profiles.keys()
    true_labels = profiles.regroup(true_group_name)
    profiles.data = np.array([d for k, d in zip(keys, profiles.data) if tuple(k) in true_labels])
    profiles._keys = [k for k in keys if tuple(k) in true_labels]
    keys = profiles.keys()
    labels = list(set(true_labels.values()))

    if holdout_group_name:
        holdouts = profiles.regroup(holdout_group_name)
    else:
        holdouts = dict((k, k) for k in keys)

    confusion = {}
    for ho in set(holdouts.values()):
        test_set_mask = np.array([tuple(holdouts[k]) == ho for k in keys], 
                                 dtype=bool)
        training_features = profiles.data[~test_set_mask, :]
        training_labels = [labels.index(true_labels[tuple(k)]) 
                           for k, m in zip(keys, ~test_set_mask) if m]

        model = train(training_features, training_labels, distance=distance)
        for k, f, m in zip(keys, profiles.data, test_set_mask):
            if not m:
                continue
            true = true_labels[k]
            predicted = labels[model.classify(f)]
            confusion[true, predicted] = confusion.get((true, predicted), 0) + 1
    return confusion

def print_confusion_matrix(confusion):
   cm = confusion_matrix(confusion)
   print cm
   print 'Overall: %d / %d = %.0f %%' % (np.diag(cm).sum(), cm.sum(),
                                         100.0 * np.diag(cm).sum() / cm.sum())

def read_ground_distance(filename, classes):
    d = np.zeros((len(classes), len(classes)))
    with open(filename) as f:
        reader = csv.reader(f)
        header = reader.next()
        if header != ['predicted', 'actual', 'distance']:
            raise RuntimeError('%s: Header line not as expected' % filename)
        for i, row in enumerate(reader):
            predicted, actual, distance = row
            if predicted not in classes:
                raise RuntimeError('%s:%d: Unknown class: %s' % (filename, i + 1, predicted))
            if actual not in classes:
                raise RuntimeError('%s:%d: Unknown class: %s' % (filename, i + 1, actual))
            d[classes.index(predicted), classes.index(actual)] = float(distance)
    return d

if __name__ == '__main__':
    parser = OptionParser("usage: %prog [-c] [-h HOLDOUT-GROUP] PROPERTIES-FILE PROFILES-FILENAME TRUE-GROUP")
    parser.add_option('-c', dest='csv', help='input as CSV', action='store_true')
    parser.add_option('-d', dest='distance', help='distance metric', default='cosine', action='store')
    parser.add_option('-g', dest='ground_distance_file', help='ground distance file', action='store')
    parser.add_option('-H', dest='holdout_group', help='hold out all that map to the same holdout group', action='store')
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('Incorrect number of arguments')
    properties_file, profiles_filename, true_group_name = args
    if options.ground_distance_file and options.distance != 'emd':
        parser.error('The option -g can only be used with -d emd.')
    if options.distance == 'emd' and options.ground_distance_file is None:
        parser.error('The -g option is required when using -d emd.')
    cpa.properties.LoadFile(properties_file)

    if options.csv:
       profiles = Profiles.load_csv(profiles_filename)
    else:
       profiles = Profiles.load(profiles_filename)

    if options.ground_distance_file:
        ground_distance = read_ground_distance(options.ground_distance_file, profiles.variables)

    if False:
        import pylab
        pylab.matshow(ground_distance)
        pylab.colorbar()
        pylab.savefig('../results/HEAD/ground_distance.png')

    if options.distance == 'emd':
        distance = lambda a, b: emd(a, b, ground_distance)
    else:
        distance = {'cosine': cosine, 'euclidean': euclidean,
                    'cityblock': cityblock}[options.distance]

    confusion = crossvalidate(profiles, true_group_name, options.holdout_group,
                              distance=distance)
    write_confusion(confusion, sys.stdout)
