'''
Krh3rtz 2017

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
	

El uso de los clasificadores de tipo árbol es utilizado por su
fácil interpretación con base a los datos. En la práctica es
muy fácil de utilizar.

Iris es un problema de Machine Learning clásico que nos permite
identificar de qué flor hablamos basándonos en en diferentes
medidas por ejemplo el ancho de los pétalos de la flor.

En este ejemplo se dan 150 características corespondientes a 3 tipos
de flor Iris.

1.- Vamos a importar un dataset.
2.- Entrenaremos al clasificador.
3.- Predeciremos en caso de no conocer una nueva flor.

ref: http://scikit-learn.org/stable/datasets/?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt

# Es posible imprimir tanto los nombres (etiquetas) o las características
print (iris.feature_names)
print (iris.target_names)
print (iris. data[0])

En caso de hacer uso de una versión antigüa de Python instala

	pip install pydot

Para una versión nueva

	pip3 install pydotplus

Graphviz tiene que ser instalado en el sistema 

	aptitude install graphviz
	pip3 install graphviz

Los módulos a instalar son:

	pip3 install sklearn scipy numpy graphviz 
	
Código probado en Python >= 3

'''
# Para crear el árbol de desiciones
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Para generar un pdf del árbol de desiciones
import pydotplus
from sklearn.externals.six import StringIO

iris = load_iris()
test_idx = [0, 50, 100]

# Training data
# Borrar las posiciones 0, 50 y 100 de los targets y data de iris -> Matríz con mayor parte de información
train_target = np.delete (iris.target, test_idx)
train_data = np.delete (iris.data, test_idx, axis=0)

# Testing data -> matríz con los ejemplos eliminados a utilizar para alimentar al clasificador.
test_target = iris.target [test_idx]
test_data = iris.data [test_idx]

# Entrenar con datos
clasificador = tree.DecisionTreeClassifier ()
clasificador.fit (train_data, train_target)
 
print (test_target)

# Indicamos la data para que nos regrese etiquetas o las flores.
print (clasificador.predict (test_data))	

# Generar un pdf con el árbol de flujo
dot_data = StringIO ()
tree.export_graphviz(clasificador,
	out_file=dot_data,
	feature_names=iris.feature_names,
	class_names=iris.target_names,
	filled=True, rounded=True,
	impurity=False)

graph = pydotplus.graph_from_dot_data (dot_data.getvalue ())
graph.write_pdf ("iris.pdf")

# Visualización de las tablas  con las que alimentamos al clasificador
print (test_data[0], test_target[0])
# cabeceras de las tablas ('nombres de las columnas')
print (iris.feature_names, iris.target_names)

'''

En este ejemplo verificamos qué tan buena es la predicción de flores con valores
nuevos. Nos dimos cuenta que partiendo de nuevos datos, el clasificador
pudo encontrar un tipo de flor (0,1,2).

Esos datos fueron primeramente extraídos de las matrices 
para luego alimentar al clasificador con esa información,
probando que hace una buena predicción.

El código está basado en los ejemplos del sitio:

http://scikit-learn.org/stable/


By: Krh3rtz

Happy Hacking

__Exploiting vulnerabilities, creating new ways through__ 

'''
