import keras
from sklearn.metrics import roc_auc_score
auc_score = 0.0

class Histories(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.aucs = []
		#self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		global auc_score
		print("AUC:",auc_score)
		return

	def on_epoch_end(self, epoch, logs={}):
		global auc_score
		#self.losses.append(logs.get('loss'))
		y_pred = self.model.predict(self.model.validation_data[0])
		auc_score  = roc_auc_score(self.model.validation_data[1], y_pred)
		self.aucs.append(auc_score)		
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
                return

    
