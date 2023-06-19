# GREEN IA
===============

## Introduction et compréhension du problème
---------------

En partant d'un model séquentiel d'une IA donné, l'objectif est de trouver des solutions d'optimisation pour :
- réduire les pertes,
- améliorer la précision du modèle et sa vitesse d'exécution.


**Au départ**
Epoch 10/10
1563/1563 [==============================] - 21s 13ms/step - loss: 0.6459 - accuracy: 0.7730

## Techniques d'optimisation utilisées

## Mixed_Float16
Le passage de Float32 à Float16 est une technique d'optimsation simple qui peut accélérer la formation sur certains GPU et TPU.

```
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### BatchNormalization()
En insérant les couches BatchNormalization() après les couches Conv2D ou Dense, le modèle effectuera une normalisation par lots sur les sorties de ces couches. La normalisation par lots permet de normaliser les activations de la couche précédente, ce qui rend le processus de formation plus stable et plus rapide. Elle permet d'améliorer la capacité de généralisation du modèle et atténuer l'adaptation excessive.

*Résultat :* En utilisant cette technique, nous arrivons à augmenter la précision de notre modèle et à réduire les pertes.

*Exemple :*
early_stopping = EarlyStopping(
    monitor='val_accuracy',   # Monitor validation accuracy
    mode='max',               # Maximize the monitored quantity
    patience=2               # Number of epochs with no improvement before stopping
)

### Early_stopping
L'ajout du Early_Stopping en tant que callback dans la fonction .fit() appliquée à notre permet de configurer de manière plus précise les paramètres de notre modèle avec des arrêts anticipés. L'arrêt précoce est une technique de régularisation qui arrête l'entraînement si, par exemple, la perte de validation atteint un certain seuil.

*Exemple :*

```
early_stopping = EarlyStopping(
    monitor='val_accuracy',   # Monitor validation accuracy
    mode='max',               # Maximize the monitored quantity
    patience=2               # Number of epochs with no improvement before stopping
)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[early_stopping])
```

*Résultat :* Cette optimisation permet de gagner du temps dans la vitesse d'exécution.


## Performance du modèle
En accumulant toutes ces techniques d'optimisation, le modèle est optimisé avec loss: 0.4218 - accuracy: 0.8520 au Epoch10

## Conclusion
Ces techniques permettent d'améliorer le modèle. D'autres techniques peuvent être envisagées comme notamment le choix de l'optimizer.

**Au final**
Epoch 10/10
1563/1563 [==============================] - 32s 20ms/step - loss: 0.4218 - accuracy: 0.8520
