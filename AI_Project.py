import mlrose
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.metrics import confusion_matrix
startTime = datetime.now()

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

# reading in the csv as a dataframe
import pandas as pd
df = pd.read_csv('EYES.csv')

# selecting the features and target
X = df[eyes]
y = df['truth_value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

iters = 50
initial_acc = 50
acc = 0

def random_hill_climb(iterations): 

    nn_model = mlrose.NeuralNetwork(
        hidden_nodes = [4],
        activation = 'relu',
        algorithm = 'random_hill_climb',
        max_iters = iterations,
        is_classifier = True,
        learning_rate=0.00001,
        early_stopping=True,
        clip_max=5,
        random_state = 156)
        
    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    y_test_accuracy1=accuracy_score(y_test,y_test_pred)
    return(y_test_pred,y_test_accuracy1)

def simulated_annealing(iterations): 

    nn_model = mlrose.NeuralNetwork(
        hidden_nodes = [4],
        activation = 'relu',
        algorithm = 'simulated_annealing',
        max_iters = iterations,
        is_classifier = True,
        learning_rate=0.00001,
        early_stopping=True,
        clip_max=5,
        random_state = 156)
        
    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    y_test_accuracy2=accuracy_score(y_test,y_test_pred)
    return(y_test_pred,y_test_accuracy2)


        
def genetic_alg(iterations): 

    nn_model = mlrose.NeuralNetwork(
        hidden_nodes = [4],
        activation = 'relu',
        algorithm = 'genetic_alg',
        max_iters = iterations,
        is_classifier = True,
        learning_rate=0.00001,
        early_stopping=True,
        clip_max=5,
        random_state = 156)
        
    nn_model.fit(X_train, y_train)
    y_test_pred = nn_model.predict(X_test)
    y_test_accuracy2=accuracy_score(y_test,y_test_pred)
    return(y_test_pred,y_test_accuracy2)



df_results = []
acc1 = random_hill_climb(iters)
acc2 = simulated_annealing(iters)
acc3 = genetic_alg(iters)
df_results.append(['random_hill_climb',acc1[1]])
df_results.append(['simulated_annealing',acc2[1]])
df_results.append(['genetic_alg',acc3[1]])

df_results=sorted(df_results,key=lambda x:x[1])
max_acc_algo=df_results[0][0]
max_acc_value=df_results[0][1]


if max_acc_algo == 'random_hill_climb':
    model = 1
elif max_acc_algo == 'simulated_annealing':
    model = 2
else:
    model = 3

acc_val = max_acc_value


i = 1

while (i < 2 and acc_val < 97):
    iters = iters + 150
    
    if model == 1:
        y_test_accuracy = random_hill_climb(iters)
    elif model ==2:
        y_test_accuracy = simulated_annealing(iters)
    else:
        y_test_accuracy = genetic_alg(iters)

    if(y_test_accuracy[1] * 100 == acc_val):
        i+=1
    else:
        i=0
    acc_val = y_test_accuracy[1] * 100

cm=confusion_matrix(y_test,y_test_accuracy[0])
print(cm)

print("The final accuracy is:",round(acc_val,4))
print("Total iterations:",iters)
print(model)
print("Execution time in seconds = ",datetime.now()-startTime,"\n")

 
