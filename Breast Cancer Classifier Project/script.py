import seaborn
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()
# print(breast_cancer_data.data, breast_cancer_data.feature_names)
# print(breast_cancer_data.target, breast_cancer_data.target_names)

from sklearn.model_selection import train_test_split

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

# print(len(training_data),len(training_labels))

from sklearn.neighbors import KNeighborsClassifier
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels)*100)

import matplotlib.pyplot as plt
k_list = range(1, 101)
plt.plot(k_list, accuracies, "b-")
plt.xlabel("Values of K")
plt.ylabel("Validation Accuracy (in %)")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()