import csv
import matplotlib.pyplot as plt

# verilerin yüklenmesi
with open('urunler.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

# verilerin fiyat aralıklarına göre gruplanması
groups = {
    '20-30 TL': [],
    '30-60 TL': [],
    '61 TL ve Üzeri': []
}
# verilerin ekrana yazdırılması
print(data)
# verilerin gruplanması
for item in data:
    price = item.get('price')
    if price is None:
        print("Fiyat bilgisi bulunamadı:", item)
    elif 20 <= int(price) < 30:
        groups['20-30 TL'].append(item)
    elif 30 <= int(price) < 61:
        groups['30-60 TL'].append(item)
    else:
        groups['61 TL ve Üzeri'].append(item)

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
