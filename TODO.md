#TODO

## Objetivo
* El objetivo de esta tarea es implementar un método de búsqueda de imágenes donde el usuario puede describir con 
palabras la escena que busca y el sistema localiza dentro de una base de datos de imágenes sin etiquetar las que mejor 
cumplen con la descripción dada.

## TODO 
* Fase 1: entrenamiento

  1.1- [DONE] Implementar un método que calcule un descriptor de texto para cada frase en español. 
  1.2- [DOING] Utilizando los datos de entrenamiento, deberá entrenar un modelo de regresión basado en redes neuronales que para
   cada descriptor de texto de entrada pueda predecir el descriptor visual de la imagen correspondiente.

* Fase 2: búsqueda

  2.1- [DOING] El método de búsqueda recibe una frase en español de consulta del usuario a la que deberá calcular su descriptor 
  de texto. 
  2.2- [ ] Utilizar el modelo de regresión ya entrenado con el descriptor de texto como entrada y obtener como salida una 
  predicción de descriptor visual. 
  2.3- [ ] Con el descriptor visual generado deberá buscar entre las imágenes existentes en la colección la que tiene el 
  descriptor visual más cercano a la predicción y escoger esa imagen como resultado de la consulta.

* Fase 3: evaluación
    ** Para evaluar la efectividad de su método deberá utilizar las 5.000 frases del conjunto de test como consulta
  3.1- [ ] Al realizar la búsqueda de los más cercanos, determine la posición (rank) en la que aparece la respuesta 
  correcta entre las 1.000 imágenes de la colección.
       3.1.1- [ ] Ordene todos los descriptores visuales del más cercano al más lejano a la predicción y localice la posición
        en que se encuentra el descriptor de la imagen a la que le corresponde a la frase de consulta.
  3.2- [ ] Construya un reporte de Jupyter Notebook (.ipynb) donde evalúe su modelo de búsqueda sobre el conjunto de test. 
  Para su evaluación debe calcular el rank de la respuesta correcta para las 5.000 consultas y grafíquelos en un histograma.
  Calcule indicadores de efectividad incluyendo: posición promedio, recall at 1, 5 y 10, y Mean Reciprocal Rank2 (MRR).
  3.3- [ ] En el reporte debe mostrar la evaluación de (al menos) dos variantes del cálculo del descriptor de texto 
  (a su elección). El reporte debe comparar los resultados obtenidos por ambas variantes y concluir cuál es la mejor.