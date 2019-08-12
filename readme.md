# HMM classifier

Hidden Markov Model (HMM) is a statistical Markov model 
in which the system being modeled is assumed to be a 
Markov process with unobservable (i.e. hidden) states. 

HMM-classifier is hmmlearn mini-wrapper 
for classification problems. It build separate HMMs for 
every label. 

##### Requirements 
```
scipy
hmmlearn
numpy
```
##### Installation 
```
pip install hmm-classifier
```

##### Based on hmmlearn project
https://hmmlearn.readthedocs.io/en/latest/

Supported HMMs:
* MultinomialHMM - Hidden Markov Model with multinomial 
(discrete) emissions
* GaussianHMM - Hidden Markov Model with 
Gaussian (continues) emissions.
 

##### Example
```
from hmmlearn import hmm
import numpy as np
from hmm_classifier import HMM_classifier

x = np.random.randint(0, 10, size=(300, 10, 2))
y = np.random.randint(0, 10, size=(300))

model = HMM_classifier(hmm.MultinomialHMM())
model.fit(x,y)

# Predict probability per label
pred = model.predict_proba(np.random.randint(0, 10, size=(10, 2)))

# Get label with the most high probability
pred = model.predict(np.random.randint(0, 10, size=(100, 2)))
```


