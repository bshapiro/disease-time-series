from pomegranate import *
import numpy as np
from load_data import load_data
from khmm import *
"""
d1 = NormalDistribution(-1, 1)
d2 = NormalDistribution(-.5, .5)
d3 = NormalDistribution(0, .5)

s1 = State( d1, name="s1" )
s2 = State( d2, name="s2" )
s3 = State( d3, name="s3" )

model1 = HiddenMarkovModel('example')
model1.add_states([s1, s2, s3])
model1.add_transition(model1.start, s1, 0.90 )
model1.add_transition(model1.start, s2, 0.10 )
model1.add_transition(s1, s1, 0.80 )
model1.add_transition(s1, s2, 0.20 )
model1.add_transition(s2, s2, 0.90 )
model1.add_transition(s2, s3, 0.10 )
model1.add_transition(s3, s3, 0.70 )
model1.add_transition(s3, model1.end, 0.30 )
model1.bake()

d1 = NormalDistribution(1, 1)
d2 = NormalDistribution(5, .5)
d3 = NormalDistribution(0, .5)

s1 = State( d1, name="s1" )
s2 = State( d2, name="s2" )
s3 = State( d3, name="s3" )

model2 = HiddenMarkovModel('example')
model2.add_states([s1, s2, s3])
model2.add_transition(model2.start, s1, 0.90 )
model2.add_transition(model2.start, s2, 0.10 )
model2.add_transition(s1, s1, 0.80 )
model2.add_transition(s1, s2, 0.20 )
model2.add_transition(s2, s2, 0.90 )
model2.add_transition(s2, s3, 0.10 )
model2.add_transition(s3, s3, 0.70 )
model2.add_transition(s3, model2.end, 0.30 )
model2.bake()
"""
gc, mt, track = load_data(100)
data, labels = df_to_sequence_list(gc.data)
assignments = np.array([i % 3 for i in range(labels.size)])
model1 = init_gaussian_hmm(data[np.where(assignments == 0)[0], :], 3, '1')
model2 = init_gaussian_hmm(data[np.where(assignments == 1)[0], :], 3, '2')
model3 = init_gaussian_hmm(data[np.where(assignments == 2)[0], :], 3, '3')

fixed = np.array([0] * 1000)
eps = 1e-3
max_iter = 50
save_name = 'test.txt'
#import pdb; pdb.set_trace()
cluster(models=np.array([model1, model2, model3]), noise_models=np.array([]),
        sequences=data, assignments=assignments, labels=labels, fixed=fixed,
        eps=eps, max_it=max_iter, odir='test')

#print model.log_probability(data)