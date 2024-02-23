import cv2
import os

path = 'ImagesQuery'
orb = cv2.ORB_create(nfeatures = 5000) # Se manda a llamar al algoritmo ORB
images = []
classNames = []
myList = os.listdir(path)

# Generador de directorios
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
    
# Obtiene los descriptores de cada imagen
def findDescriptors(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    
    return(desList)

# Obtiene a que imagen se parece mas
def findID(img, desList, params,threshold=5):
    kp2, des2 = orb.detectAndCompute(img, None)
    search_params = {}
    #bf = cv2.BFMatcher()
    flann = cv2.FlannBasedMatcher(params, search_params)
    matchList = []
    finalVal = -1
    
    try:
        for des in desList:
            #matches = bf.knnMatch(des, des2, k = 2)
            matches = flann.knnMatch(des, des2, k=2)
            good = []
                
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append([m])
                
            matchList.append(len(good))
    except:
        pass
    
    if len(matchList) != 0:
        if max(matchList) > threshold:
            finalVal = matchList.index(max(matchList))
            
    print(matchList)
    return finalVal
            
desList = findDescriptors(images)

# Testeo de imagenes

# Imagenes de testeo: Videojuego.jpg, Doritos.jpg, Coca1.jpg, Coca2.jpg, Coca3.jpg
img = cv2.imread('Photos/Videojuego.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

index_params = dict(algorithm=6, 
                    trees=10,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2)

id = findID(imgGray, desList, index_params)

if id != -1:
    kp1, des1 = orb.detectAndCompute(images[id], None)
    kp2, des2 = orb.detectAndCompute(imgGray, None)
    
    search_params = {}
    #bf = cv2.BFMatcher()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
        
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])
            
    img3 = cv2.drawMatchesKnn(images[id], kp1, imgGray, kp2, good, None, flags = 2)
    img3 = cv2.resize(img3, None, fx=0.5, fy=0.5)
    cv2.imshow('Matches', img3)
    cv2.putText(img, classNames[id], (75, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
else:
    cv2.putText(img, 'No match', (75, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
    
img = cv2.resize(img, None, fx=0.5, fy=0.5)
    
cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()