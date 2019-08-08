from hmm_classifier import GHMM_classifier
import numpy as np

x = np.random.randint(0, 100, size=(300, 10, 2))

y = np.random.randint(0, 10, size=(300))

print(x)
print(y)


model = GHMM_classifier(n_components=3)
model.fit(x,y)

pred = model.predict_proba(np.random.randint(0, 100, size=(10, 2)))
print(pred)

pred = model.predict(np.random.randint(0, 100, size=(100, 2)))
print(pred)