# Clasificador de imágenes SVM

![N|Solid](https://ibin.co/3oqm9iSzggjw.png)

La aplicación ha sido creada usando python 3.4, interfaz gráfica se ha construido usando PySide. Otras dependencias son:
  - numpy: para el manejo de arrays.
  - pandas: para la lectura de imágenes.
  - sklearn: para el entrenamientos del SVM.

# Uso de la aplicación
Una vez que se tienen todos lo módulos intalados, la aplicación se puede ejecutar con el comando:
    
```sh
$ python3.4 app.py
```
Una vez la aplicación ha sido ejecutada se abrirá la siguiente ventana:

![N|Solid](https://ibin.co/3oqoXhvkwuc6.png)

Se observa que todos los botones están desactivado. Lo primero que hay que hacer es seleccionar la carpeta donde se encuentran las imágenes a clasificar, para ello se pulsa en Menú-> Abrir Train Dir (ver siguiente figura).
![N|Solid](https://ibin.co/3oqmRpCdDead.png)

Tras seleccionar la carpeta donde se encuentran los ficheros `.dat` se habilitará el botón de entrenar. En este punto se debe seleccionar el Kernel que se va a usar y los parámetros que este use.

Tras pulsar entrenar y si el proceso acaba correctamente se habilitan los botones de Clasificar y Clasificar Carpeta. El funcionamiento de estos botones es el siguiente.
* Clasificar: permite seleccionar una sola imagen.
* Clasificar Carpeta: permite seleccionar una carpeta donde haya varias imágenes.

Tras realizar la clasificación se habilita el botón de  Ver detalles. Este botón muestra el resultado de la clasificación. Permite comparar la predicción realizada con el SVM y la etiqueta real de la imagen. En la siguiente figura se muestra un ejemplo:
![N|Solid](https://ibin.co/3oqmKKcdxjVw.png)

Este es el resultado de entrenar el clasificador para reconocer dígitos escritos a mano, se observa que tiene una precisión del 89%, para cada dígito por separado se tienen los siguientes datos:
> Dígito 1 (data): precisión del 39%
> Dígito 0 (data0): precisión del 91%
> Dígito 2 (data2): precisión del 91%
> Dígito 3 (data3): precisión del 100%
> Dígito 4 (data4): precisión del 100%
> Dígito 5 (data5): precisión del 100%
> Dígito 6 (data6): precisión del 88%
> Dígito 7 (data7): precisión del 90%
> Dígito 8 (data8): precisión del 100%

Dependiendo del número de imágenes de entrenamiento y su tamaño, el tiempo de entrenamiento puede ser largo, por ello se ha habilitado una opción que permite guardar clasificadores ya entrenados y cargar uno ya entrenado. Esta opción se encuentra en Menú->SMV->Cargar SMV o en Menú->SMV->Guardar SMV. Los clasificadores guardados se deben almacenar en formato `.pkl`.

El clasificador se pueden entrenar usando distintos kernel que son los siguientes:
  - 'Linear': no usa parámetros.
  - 'Polynomial': usa los parámetros degree y coef0.
  - 'RBF': usa el parámetro gamma.
  - 'Sigmoid': usa el parámetro coef0.

El kernel Polynomial no funciona correctamente para degree >=10 por lo que se ha limitado el rango de valores que puede tener este parámetro.

Para entrenar el clasificador es necesario que las imágenes tengan un formato concreto:

> 1. Formato .dat
> 2. Imágenes de un solo canal (no usar imágenes RGB)
> 3. El fichero debe estar compuesto por 3 columnas donde las dos primeras son la posición x,y del pixel y la tercera la intensidad.


#### Autor

La aplicación ha ido escrita por Gustavo Plaza Roma para la asignatura de Minería de Datos del MÁSTER UNIVERSITARIO EN INGENIERÍA DE SISTEMAS Y DE CONTROL de la UNED.
