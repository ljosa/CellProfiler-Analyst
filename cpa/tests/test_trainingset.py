import os
import cpa
from cpa.trainingset import TrainingSet

def test_init():
    cpa.properties.LoadFile(os.getenv('CPA_PROPERTIES'))
    ts = TrainingSet()

def test_one_vs_all():
    cpa.properties.LoadFile(os.getenv('CPA_PROPERTIES'))
    ts = TrainingSet('example_threeclass.txt')
    training_sets = ts.one_vs_all()
    assert len(training_sets) == 3
