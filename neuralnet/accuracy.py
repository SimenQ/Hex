import matplotlib.pyplot as plt
from keras.callbacks import Callback

class PlotAccuracyAndSaveModelCallback(Callback):
    def __init__(self, model_path):
        self.accuracy = []
        self.best_accuracy = 0
        self.best_model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        self.accuracy.append(logs['accuracy'])
        if logs['accuracy'] > self.best_accuracy:
            self.best_accuracy = logs['accuracy']
            self.model.save(self.best_model_path)

    def on_train_end(self, logs=None):
        plt.plot(self.accuracy)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
