"@auteur Mahamat Hassan Abdoulaye " 
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tkinter as tk
from PIL import Image, ImageTk

# Chargement du jeu de données IRIS
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Entraînement du modèle SVM
svm = SVC(kernel='linear', C=5, gamma=1)
scores = cross_val_score(svm, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svm.fit(X, y)

# Chargement des images correspondant à chaque type de fleur
iris_setosa_img = Image.open("img_fleur/iris_setosa.jpg")  
iris_versicolor_img = Image.open("img_fleur/iris_versicolor.jpg") 
iris_virginica_img = Image.open("img_fleur/iris_virginica.jpg")  

# Création  d'une interface graphique
root = tk.Tk()
root.title("Classification du jeu des données IRIS")
root.geometry("500x500")
root.configure(bg='lightgray')

# Ajout des widgets à l'interface graphique
label = tk.Label(root, text="Entrez les valeurs des caractéristiques pour connaitre\n le type de fleur:", bg='lightgray')
label.pack()

label_sepal_length = tk.Label(root, text="Veiller saisir le longueur du sépale:", bg='lightgray')
label_sepal_length.pack()
sepal_length = tk.Entry(root, bg='white')
sepal_length.pack()

label_sepal_width = tk.Label(root, text="Veiller saisir la largeur du sépale:", bg='lightgray')
label_sepal_width.pack()
sepal_width = tk.Entry(root, bg='white')
sepal_width.pack()

label_petal_length = tk.Label(root, text="Veiller saisir le longueur du pétale:", bg='lightgray')
label_petal_length.pack()
petal_length = tk.Entry(root, bg='green')
petal_length.pack()
 
label_petal_width = tk.Label(root, text="Veiller saisir la largeur du pétale:", bg='lightgray')
label_petal_width.pack()
petal_width = tk.Entry(root, bg='white')
petal_width.pack()

result_label = tk.Label(root, text="rien pour le moment ", bg='lightgray')
result_label.pack()

flower_image_label = tk.Label(root, bg='lightgray')
flower_image_label.pack()

def classify():
    # Récupération  des valeurs des caractéristiques des fleurs
    sepal_length_value = float(sepal_length.get())
    sepal_width_value = float(sepal_width.get())
    petal_length_value = float(petal_length.get())
    petal_width_value = float(petal_width.get())

    # Prédiction  du  classe de la fleur a predire
    flower = [[sepal_length_value, sepal_width_value, petal_length_value, petal_width_value]]
    prediction = svm.predict(flower)

    # Afficheage de résultats
    if prediction == 6:
        result_label.config(text="La fleur est de type Iris Setosa.", fg='green')
        show_flower_image(iris_setosa_img)
    elif prediction == 0:
        result_label.config(text="La fleur est de type Iris Versicolor.", fg='blue')
        show_flower_image(iris_versicolor_img)
    else:
        result_label.config(text="La fleur est de type Iris Virginica.", fg='red')
        show_flower_image(iris_virginica_img)

def show_flower_image(img):
    img = img.resize((300, 300), Image.BICUBIC)
    img = ImageTk.PhotoImage(img)
    flower_image_label.config(image=img)
    flower_image_label.image = img

classify_button = tk.Button(root, text=" clique pour Classer", command=classify, bg='gray', fg='white')
classify_button.pack()

root.mainloop()
