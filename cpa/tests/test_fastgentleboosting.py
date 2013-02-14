import os
from numpy import array
import cpa
from cpa.trainingset import TrainingSet
from cpa.fastgentleboosting import FastGentleBoostingMulticlass, FastGentleBoostingOneVsAll

def load_26class():
    cpa.properties.LoadFile(os.getenv('CPA_PROPERTIES'))
    return TrainingSet('example_threeclass.txt')

MULTICLASS_MODEL = [('Nuclei_Intensity_pH3_IntegratedIntensityEdge', 2.62819, array([ 0.87755175, -0.87755175]), array([-1.00000019,  1.00000019]), 0.96739130453004141), ('Cells_Texture_3_pH3_GaborX', 4.0923600000000002, array([ 0.55045042, -0.55045042]), array([-1.,  1.]), 0.96546256905232852), ('Nuclei_AreaShape_MinorAxisLength', 12.1145, array([ 0.61327406, -0.61327406]), array([-1.,  1.]), 0.94285600398270775), ('Nuclei_Texture_1_pH3_GaborX', 0.88371599999999995, array([ 0.76551232, -0.76551232]), array([-0.7880939,  0.7880939]), 0.91944105140355659), ('Cytoplasm_AreaShape_Solidity', 0.60338999999999998, array([-0.67699131,  0.67699131]), array([ 0.839865, -0.839865]), 0.88703342428163712)]
ONEVSALL_MODEL = [[('Nuclei_Intensity_pH3_IntegratedIntensityEdge', 2.62819, array([ 0.87755175, -0.87755175]), array([-1.00000019,  1.00000019]), 0.96739130453004141), ('Cells_Texture_3_pH3_GaborX', 4.0923600000000002, array([ 0.55045042, -0.55045042]), array([-1.,  1.]), 0.96546256905232852), ('Nuclei_AreaShape_MinorAxisLength', 12.1145, array([ 0.61327406, -0.61327406]), array([-1.,  1.]), 0.94285600398270775), ('Nuclei_Texture_1_pH3_GaborX', 0.88371599999999995, array([ 0.76551232, -0.76551232]), array([-0.7880939,  0.7880939]), 0.91944105140355659), ('Cytoplasm_AreaShape_Solidity', 0.60338999999999998, array([-0.67699131,  0.67699131]), array([ 0.839865, -0.839865]), 0.88703342428163712)], [('Cells_Correlation_Correlation_DNA_and_pH3', 0.63361800000000001, array([-0.99999883,  0.99999883]), array([ 0.71428548, -0.71428548]), 0.91666666744276881), ('Nuclei_Intensity_pH3_IntegratedIntensity', 10.0806, array([-1.,  1.]), array([ 0.42307605, -0.42307605]), 0.92320894385104224), ('Cytoplasm_Texture_3_pH3_GaborY', 0.21675800000000001, array([-0.68539643,  0.68539643]), array([ 0.5502097, -0.5502097]), 0.89340492216670175), ('Nuclei_AreaShape_Zernike7_5', 3.7617400000000001, array([-0.88110986,  0.88110986]), array([ 0.47161638, -0.47161638]), 0.86227335164491037), ('Nuclei_Intensity_DNA_MaxIntensity', 0.61474799999999996, array([-0.76947775,  0.76947775]), array([ 0.46671652, -0.46671652]), 0.86159910182657362)], [('Nuclei_AreaShape_Zernike6_4', 1.44817, array([ 0.38775506, -0.38775506]), array([-0.61290331,  0.61290331]), 0.73749999916180975), ('Cells_AreaShape_Zernike9_9', 2.3670900000000001, array([-0.57893593,  0.57893593]), array([ 0.45673762, -0.45673762]), 0.73573210774215814), ('Cytoplasm_Texture_1_Actin_GaborY', 0.149671, array([-0.49736884,  0.49736884]), array([ 0.49515324, -0.49515324]), 0.69255571265580407), ('AreaNormalized_Cells_AreaShape_Zernike4_2', 0.012863100000000001, array([ 0.21643471, -0.21643471]), array([-1.,  1.]), 0.70789117367063814), ('Nuclei_Texture_1_DNA_GaborX', 1.7382599999999999, array([ 0.47937057, -0.47937057]), array([-0.48302864,  0.48302864]), 0.69507299952045831)]]

def test_multiclass_ParseModel():
    algorithm = FastGentleBoostingMulticlass()
    algorithm.model = MULTICLASS_MODEL
    s = algorithm.ShowModel()
    algorithm.ParseModel(s)

def test_multiclass_ShowModel():
    algorithm = FastGentleBoostingMulticlass()
    algorithm.model = MULTICLASS_MODEL
    algorithm.ShowModel()

def test_multiclass_Train():
    ts = load_26class()
    algorithm = FastGentleBoostingMulticlass()
    algorithm.Train(ts, 5)

def test_onevsall_ParseModel():
    algorithm = FastGentleBoostingOneVsAll()
    algorithm.model = ONEVSALL_MODEL
    s = algorithm.ShowModel()
    algorithm.ParseModel(s)

def test_onevsall_ShowModel():
    algorithm = FastGentleBoostingOneVsAll()
    algorithm.model = ONEVSALL_MODEL
    algorithm.ShowModel()

def test_onevsall_Train():
    ts = load_26class()
    algorithm = FastGentleBoostingOneVsAll()
    algorithm.Train(ts, 5)
