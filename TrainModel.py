import os
import cv2
import numpy as np
from sklearn import svm, model_selection
from skimage.feature import hog 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import image as mpimg 
from matplotlib import pyplot as plt
import joblib

# Extraer imagenes de prueba 

def get_files(directory):
    all_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
            
    return all_files

# Dataset de cada objeto y dataset de muestras negativas

train_coca_pos = get_files("./TrainingData/Coca")
train_doritos_pos = get_files("./TrainingData/Doritos")
train_videojuego_pos = get_files("./TrainingData/Videojuego")
train_neg = get_files("./TrainingData/NegativeSamples")

# Generacion de modelo

win_size = (16, 16)
block_size = (8, 8)
block_stride = (4, 4)
cell_size = (4, 4)
nbins = 9

# Se preparan las muestras positivas para el videojuego

samples = []
labels = []

for image_path in train_videojuego_pos:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=nbins, pixels_per_cell=cell_size,
                   cells_per_block=block_size, block_norm='L2-Hys', visualize=False)
    samples.append(features)
    labels.append(1)
    
# Se preparan las muestras positivas para la coca

for image_path in train_coca_pos:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=nbins, pixels_per_cell=cell_size,
                   cells_per_block=block_size, block_norm='L2-Hys', visualize=False)
    samples.append(features)
    labels.append(2)
    
# Se preparan las muestras positivas para los doritos

for image_path in train_doritos_pos:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=nbins, pixels_per_cell=cell_size,
                   cells_per_block=block_size, block_norm='L2-Hys', visualize=False)
    samples.append(features)
    labels.append(3)
        
# Se preparan las muestras negativas

for image_path in train_neg:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=nbins, pixels_per_cell=cell_size,
                   cells_per_block=block_size, block_norm='L2-Hys', visualize=False)
    samples.append(features)
    labels.append(0)
    
# Dividir el dataset para entrenar y probar los resultados

x_train, x_test, y_train, y_test = model_selection.train_test_split(samples, labels, test_size=0.3,
                                                                    random_state=78, stratify=labels,
                                                                    shuffle=True)

# Se define el modelo

svm_model = svm.SVC()

# Se define un conjunto de hiperparametros

param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['linear', 'rbf', 'poly']}

# Se utiliza randomized search para encontrar los mejores parametros

random_search = RandomizedSearchCV(svm_model, param_grid, n_iter=10, cv=3, verbose=3)
random_search.fit(x_train, y_train)

# Se obtiene el mejor modelo

best_model = random_search.best_estimator_
#print("Los mejores parametros son: ", random_search.best_params_)

# Resultados con los datos de prueba

y_pred = best_model.predict(x_test)
#print("Reporte de clasificacion: \n", classification_report(y_test, y_pred))

# Generar el modelo final 

final_model = svm.SVC(kernel=random_search.best_params_["kernel"], C=random_search.best_params_["C"],
                      gamma=random_search.best_params_["gamma"])
final_model.fit(x_train, y_train)

# Se exporta el modelo para usarse dentro de otro codigo

joblib.dump(final_model, 'modelo_svm.pkl')

# Se imprimen los resultados

y_pred = final_model.predict(x_test)
confusion_mat = confusion_matrix(y_test, y_pred)

# 1. Carga la imagen de prueba

test_image = cv2.imread("Photos/Doritos1.jpg")
test_image = cv2.resize(test_image, (64, 128))

# 2. Extracción de características HOG de la imagen de prueba

gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

features_test = hog(gray_test_image, orientations=nbins, pixels_per_cell=cell_size,
               cells_per_block=block_size, block_norm='L2-Hys', visualize=False)

# 3. Predicción con el modelo SVM

prediction = final_model.predict([features_test])

if prediction == 1:
    print("Se encontro un videojuego")
elif prediction == 2:
    print("Se encontro una coca")
elif prediction == 3:
    print("Se encontro unos doritos")
else:
    print("El objeto de interés no está presente en la imagen.")