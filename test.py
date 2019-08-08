from hmmlearn import hmm

import numpy as np

from hmm_classifier import HMM_classifier

x = np.random.randint(0, 10, size=(300, 10, 2))

y = np.random.randint(0, 10, size=(300))

print(x)
print(y)


model = HMM_classifier(hmm.MultinomialHMM())
model.fit(x,y)

pred = model.predict_proba(np.random.randint(0, 10, size=(10, 2)))
print(pred)

pred = model.predict(np.random.randint(0, 10, size=(100, 2)))
print(pred)