import csv
import numpy as np
import tensorflow as tf
import os
# Diğer TensorFlow kodunuz burada
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with tf.device('/cpu:0'):
    # TensorFlow kodunuz burada

    # CSV dosyasından verilerin okunması
    data = []
    with open('health-data.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([
                float(row['age']),
                float(row['weight']),
                float(row['height']),
                float(row['bmi'])
            ])

    # Verilerin ayırılması (eğitim ve test verileri)
    np.random.shuffle(data)
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    # Verilerin numpy dizilerine dönüştürülmesi
    train_inputs = np.array([x[:3] for x in train_data])
    train_labels = np.array([x[3] for x in train_data])
    test_inputs = np.array([x[:3] for x in test_data])
    test_labels = np.array([x[3] for x in test_data])

    # Yapay sinir ağı modelinin oluşturulması
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(16, input_dim=3, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Modelin eğitilmesi
    model.fit(train_inputs, train_labels, epochs=100, batch_size=8, verbose=2)

    #Modelin test edilmesi
    test_loss = model.evaluate(test_inputs, test_labels, verbose=0)
    print('Test loss:', test_loss)

    #Modelin kullanılması
    predictions = model.predict(test_inputs)
    for i in range(len(predictions)):
        print('Gerçek Değer:', test_labels[i], 'Öngörülen Değer:', predictions[i])

    #Eğitilmiş modelin kaydedilmesi
    model.save('health-model.h5')
