import csv
from sklearn.neural_network import MLPRegressor

# CSV dosyasından verilerin okunması
data = []
with open('file.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

# Ülkelerin sıralanması ve "one-hot encoding" ile sayı değerlerine dönüştürülmesi
countries = sorted(list(set([d["country"] for d in data])))
one_hot_encoding = [[1 if country == d["country"] else 0 for d in data] for country in countries]

# Giriş ve çıkış katmanlarımızı oluşturalım
X = [[int(d["championships"]), int(d["players"])] + one_hot_encoding[countries.index(d["country"]) if d["country"] in countries else 0] for d in data]

# Burada "championships", "players" ve "one-hot encoded" değerleri kullanılmıştır.
y = [int(d["goals"]) for d in data]


# Yapay sinir ağımızı eğitelim
model = MLPRegressor(max_iter=10000, verbose=True).fit(X, y)


# Tahminlerimizi yapalım
predictions = model.predict(X)

# Doğruluk oranını hesaplayalım
accuracy = sum([1 for i, prediction in enumerate(predictions) if abs(prediction - y[i]) < 0.001]) / len(predictions)



# Sonuçları yazdıralım
print(f"Accuracy: {accuracy:.2f}")
