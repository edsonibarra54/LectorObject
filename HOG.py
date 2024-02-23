import cv2
from skimage.feature import hog 
import joblib

objects = ['No se encontro ningun objeto','Videojuego','Coca cola','Doritos']

# Se importa el modelo SVM entrenado

loaded_model = joblib.load('modelo_svm.pkl')

# Es cargada la imagen y es escalada al tamaño indicado para el modelo creado

# Imagenes de testeo: Videojuego.jpg, Doritos.jpg, Coca1.jpg, Coca2.jpg, Coca3.jpg
image = cv2.imread("Photos/Coca1.jpg")
image_resized = cv2.resize(image, (64, 128))

# Parametros del modelo SVM

win_size = (16, 16)
block_size = (8, 8)
block_stride = (4, 4)
cell_size = (4, 4)
nbins = 9

# Se convierte a escala de grises la imagen

gray_image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

features = hog(gray_image_resized, orientations=nbins, pixels_per_cell=cell_size,
               cells_per_block=block_size, block_norm='L2-Hys', visualize=False)

# Predicción con el modelo SVM

prediction = loaded_model.predict([features])

# Se le añade la etiqueta a la imagen original

cv2.putText(image, objects[prediction[0]], (75, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

# Se cambie el tamaño de la imagen para mas comodidad 

image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Se muestra la imagen

cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()