'''

El uso de los clasificadores de tipo árbol es utilizado por su
fácil interpretación con base a los datos. En la práctica es
muy fácil de utilizar.

ref: https://en.wikipedia.org/wiki/Iris_flower_data_set

Iris es un problema de Machine Learning clásico que nos permite
identificar de qué flor hablamos basándonos en en diferentes
medidas por ejemplo el ancho de los pétalos de la flor.

En este ejemplo se dan 150 ejemlos de esa información.

En este caso hay cinco columnas a tratar.

1.- Vamos a importar un dataset.
2.- Entrenaremos al clasificador.
3.- Predeciremos en caso de no conocer una nueva flor.

ref: http://scikit-learn.org/stable/datasets/?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt


print (iris.feature_names)
print (iris.target_names)
print (iris. data[0])

En caso de hacer uso de una versión antigüa de python instala

pydot

Para una versión nueva

pydotplus

Graphviz tiene que ser instalado en el sistema 

	aptitude install graphviz
	pip3 install graphviz

'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# Training data
# Borrar las posiciones 0, 50 y 100 de los targets y data de iris -> Matriz con mayor parte de información
train_target = np.delete (iris.target, test_idx)
train_data = np.delete (iris.data, test_idx, axis=0)

# Testing data -> matriz con los ejemplos eliminados
test_target = iris.target [test_idx]
test_data = iris.data [test_idx]

# Entrenar con datos
clasificador = tree.DecisionTreeClassifier ()
clasificador.fit (train_data, train_target)
 
print (test_target)
# Indicamos la data para que nos regrese etiquetas o las flores
print (clasificador.predict (test_data))	


# Generar un pdf con el árbol de flujo	

from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO ()
tree.export_graphviz(clasificador,
	out_file=dot_data,
	feature_names=iris.feature_names,
	class_names=iris.target_names,
	filled=True, rounded=True,
	impurity=False)

graph = pydotplus.graph_from_dot_data (dot_data.getvalue ())
graph.write_pdf ("iris.pdf")


print (test_data[0], test_target[0])
print (iris.feature_names, iris.target_names)

'''

En este ejemplo verificamos qué tan buena es la predicción de flores con valores
nuevos. Nos dimos cuenta que partiendo de nuevos datos, el clasificador
pudo encontrar un tipo de flor (0,1,2).

Esos datos fueron primeramente extraídos de las matrices 
para luego alimentar al clasificador con esa información,
probando que hace una buena predicción.

'''