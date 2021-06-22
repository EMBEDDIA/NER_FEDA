
class EarlyStopper:

	def __init__(self, delta_loss=0.05, delta_score=0.001, patience=2):
		self.__best_loss = [float("inf"), float("inf")] #Best-delta, Best
		self.__best_score = [0.0, 0.0] #Best, Best-delta (not counted as improvement, but not penalized)

		self.__counter = [0, 0]	#No improvements, No worse no better
		self.__patience = patience
		self.__delta_loss = delta_loss
		self.__delta_score = delta_score

	def __compareLoss(self, current_loss):
		if current_loss < self.__best_loss[0]:
			return 1
		elif current_loss > self.__best_loss[1]:
			return 3
		else:
			return 2

	def __compareScore(self, current_score):
		if current_score > self.__best_score[0]:
			self.__updateScore(current_score)
			return 1
		elif current_score < self.__best_score[1]:
			return 3
		else:
			return 2

	def __updateScore(self, current_score):
		self.__best_score[0] = current_score
		self.__best_score[1] = current_score - self.__delta_score
		self.__counter = [0, 0]

	def __updateLoss(self, current_loss):
		self.__best_loss[0] = current_loss - (self.__delta_loss*current_loss)
		self.__best_loss[1] = current_loss

	def checkImprovement(self, loss, score):
		loss = loss.item()
		improvement_score = self.__compareScore(score)
		improvement_loss = self.__compareLoss(loss)
		if improvement_score == 1:
			self.__updateLoss(loss)
			return True
		elif improvement_score == 2 and improvement_loss == 1:
			self.__updateLoss(loss)
			#self.__counter[1] += 1
			return True
		elif improvement_score == 2 and improvement_loss == 2:
			self.__counter[1] += 1
			return False
		else:
			self.__counter[0] += 1
			return False

	def getCounter(self):
		return self.__counter

	def stopTraining(self):
		if self.__counter[0] > self.__patience-1:
			return True
		if self.__counter[1] > self.__patience:
			return True
		return False





