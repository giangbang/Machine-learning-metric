import numpy as np
import matplotlib.pyplot as plt

def average_precision(y_pred, y_true):
	indx = np.argsort(y_pred)
	precision, recall = [], []
	prev_pre = 0
	for i in indx[::-1]:
		prev_pre += y_true[i]
		precision.append(prev_pre)
		recall.append(prev_pre)
	
	true_label = np.sum(y_true)
	precision, recall = precision/(np.arange(len(precision))+1), recall/true_label
	precision = np.concatenate(([1], precision), axis=0)
	recall = np.concatenate(([0], recall), axis=0)
	return precision, recall
	
num_sample = 100
y_pred = np.random.uniform(size=num_sample)
y_true = np.random.randint(0, 2, size=num_sample)
precision, recall = average_precision(y_pred, y_true)

plt.subplot(1,2,1)
plt.plot(recall, precision)
a = 5
print(recall[:a])
print(precision[:a])
from sklearn.metrics import precision_recall_curve
precision, recall, thres = precision_recall_curve(y_true, y_pred)
plt.subplot(1,2,2)
plt.plot(recall, precision)
print(recall[-a+1:][::-1])
print(precision[-a+1:][::-1])
plt.show()
print(thres[:10])
print(np.sort(y_pred)[:10])