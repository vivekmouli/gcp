import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/gt_full.csv")
X = data.drop("NOX",1)
Y = data["NOX"]

# Prepare train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Save it
if not os.path.isdir("data"):
    os.mkdir("data")
np.savetxt("data/train_features.csv",X_train)
np.savetxt("data/test_features.csv",X_test)
np.savetxt("data/train_labels.csv",y_train)
np.savetxt("data/test_labels.csv",y_test)