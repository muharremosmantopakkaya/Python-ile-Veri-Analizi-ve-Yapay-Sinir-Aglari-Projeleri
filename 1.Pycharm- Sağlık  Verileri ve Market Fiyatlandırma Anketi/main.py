import matplotlib.pyplot as plt

# verilerin yüklenmesi
data = [
    {'name': 'Meyve Suyu', 'price': 5, 'type': 'Meyve Suyu', 'quantity': 100},
    {'name': 'Kola', 'price': 6, 'type': 'İçecek', 'quantity': 50},
    {'name': 'Bira', 'price': 8, 'type': 'İçecek', 'quantity': 25},
    {'name': 'Yoğurt', 'price': 4, 'type': 'Meyve Suyu', 'quantity': 150},
    {'name': 'Domates Suyu', 'price': 3, 'type': 'Meyve Suyu', 'quantity': 200}
]

# verilerin fiyat aralıklarına göre gruplanması
groups = {
    '3-4 TL': [],
    '5-6 TL': [],
    '7 TL ve Üzeri': []
}

# verilerin gruplanması
for item in data:
    if 3 <= item['price'] < 4:
        groups['3-4 TL'].append(item)
    elif 4 <= item['price'] < 7:
        groups['5-6 TL'].append(item)
    else:
        groups['7 TL ve Üzeri'].append(item)

# x ve y değerlerinin oluşturulması
x = list(groups.keys())
y = [len(group) for group in groups.values()]

# grafiğin oluşturulması
plt.plot(x, y)

# x ekseninin etiketlenmesi
plt.xlabel("Fiyat Aralıkları")

# y ekseninin etiketlenmesi
plt.ylabel("Ürün Sayısı")

# grafiğin gösterilmesi
plt.show()