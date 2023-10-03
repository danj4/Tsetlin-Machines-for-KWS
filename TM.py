from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
import pandas as pd
from time import time
import random

from sklearn.model_selection import train_test_split

X = np.load("X_trainn44.npy")
Y = np.load("Y_trainn44.npy")

print(X.shape)

X_flat = X.reshape(X.shape[0], -1) 

X_train, X_test, Y_train, Y_test = train_test_split(X_flat, Y, test_size=0.2, random_state=42)  

train_df = pd.DataFrame(data=X_train, columns=["feature_" + str(i) for i in range(X_train.shape[1])])
train_df["label"] = Y_train

test_df = pd.DataFrame(data=X_test, columns=["feature_" + str(i) for i in range(X_test.shape[1])])
test_df["label"] = Y_test

#tm = MultiClassTsetlinMachine(10000, 17, 6.0)

tm = MultiClassTsetlinMachine(
    number_of_clauses=420,
    T=17,
    s=6.0,
    boost_true_positive_feedback=1, 
    number_of_state_bits=8,  
	indexed=True,
    append_negated=True,  
    weighted_clauses=False,  
    s_range=False,  
    clause_drop_p=0.0,  
    literal_drop_p=0.0 
)

print(tm.clause_drop_p)

num_samples = X_train.shape[0]

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

 

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

 

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

 

p=[]
print(f"[",end='');
for i in range(tm.number_of_classes):
    for j in range(0, tm.number_of_clauses):

 

        l = []        
        for k in range(tm.number_of_features):
            if tm.ta_action(i, j, k) == 1:
                l.append(k)
        num_inc_ind = len(l)
        print(f"{num_inc_ind}",end=',')
        if (num_inc_ind > 0):   
            print(l, end='')


# print("\nAccuracy over 250 epochs:\n")
# for i in range(250):
# 	start_training = time()
# 	indices = np.random.permutation(num_samples)
# 	X_train = X_train[indices]
# 	Y_train = Y_train[indices]

# 	tm.fit(X_train, Y_train, epochs=1, incremental=True)
# 	stop_training = time()

# 	start_testing = time()
# 	result = 100*(tm.predict(X_test) == Y_test).mean()
# 	stop_testing = time()

# 	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))