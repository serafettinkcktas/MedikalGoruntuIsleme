import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Görseli gri tonlamalı (grayscale) olarak yükle
img = cv2.imread('adyu.png', 0)

# 2. Hızlı Fourier Dönüşümü (FFT) uygula
f = np.fft.fft2(img)

# 3. Alçak frekansları merkeze kaydır (Görselleştirme için gerekli)
fshift = np.fft.fftshift(f)

# 4. Büyüklük spektrumunu hesapla (Logaritmik ölçekte)
# log(1 + abs) kullanıyoruz çünkü değerler çok geniş bir aralıkta olabilir
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Görselleştirme
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Orijinal Görsel'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frekans Spektrumu (K-Space)'), plt.xticks([]), plt.yticks([])
plt.show()