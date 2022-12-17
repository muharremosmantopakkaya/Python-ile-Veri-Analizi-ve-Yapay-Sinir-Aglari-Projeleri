import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# CSV dosyasından verilerin okunması
X = []
y = []
with open('countries.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        X.append([int(row["population"])])
        y.append(row["country"])

# Verilerinizi eğitim ve test kümeleri olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Modelinizi oluşturun
# Modelinizi oluşturun
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
                    solver='sgd', verbose=10,  random_state=21,tol=0.000000001)


# Modelinizi eğitin
mlp.fit(X_train, y_train)

# Modelinizin performansını değerlendirin
accuracy = mlp.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
