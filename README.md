# Python ile Derin Öğrenme
Derin öğrenme, makine öğrenmesine göre daha yeni bir yaklaşımdır ve farklı bir prensibe sahiptir. Makine öğrenmesinde verilerin anlamlandırılması, geliştiricilerin
veriler içerisindeki öznitelikleri kendilerinin çıkartması sonrasında makine öğrenmesi modellerine aktarılmasıyla gerçekleştirilir. Bu sebeple modelin başarısı, büyük
ölçüde özniteliklerin (features) iyi çıkarılmış olmasına bağlı kalmaktadır. Derin öğrenmeyle birlikte bu duruma farklı bir yaklaşım getirilmiştir. Derin öğrenmenin makine öğrenmesinden en bariz farkı özniteliklerin de model tarafından öğreniliyor olmasıdır. Bu sayede karmaşık veriler derin öğrenme modelleriyle yüksek başarıyla öğrenilebilmekte
ve veriler anlamlandırılabilmektedir.


Derin öğrenme, temeli sibernetiğe dayanan insanın karar verme mekanizmalarının incelenmesi sonucu sinir hücrelerinin (nöronların) matematiksel olarak modellenmesi ve
yapay olarak taklit edilmesiyle ortaya çıkmıştır. Canlıların sinir yapısı, hiyerarşik bir şekilde sıralanmış nöronların birbirleriyle elektriksel dürtüler vasıtasıyla 
etkileşime girmesiyle oluşmaktadır. Her bir sinir hücresi bir önceki nörondan aldığı elektriksel veriyi işleyerek bağlı olduğu diğer nörona iletmektedir. Verilerin işlenmesi
sonucunda vücut kararları alınmaktadır. Bu teorik bilgiler, derin öğrenme modellerinde yapay sinir ağları (Artificial Neural Networks) olarak karşılık bulmuştur. Öncelikle
yapay nöronlar (perceptron) oluşturulmuş, daha sonra yapay sinir ağına katmanlar halinde yerleştirilmiştir. Böylece derin bir yapay öğrenme modeli gerçekleştirilmiştir.

<p align="center">
  <img src="https://github.com/mehmet-engineer/Deep_Learning_with_Python/blob/master/artificial_neural_networks.jpg" />
</p>

Her bir yapay sinir hücresinin görevi, girdi olarak aldığı verileri f(x) = Wx+b lineer fonksiyonuna göre işleyerek bir çıktı üretmesidir. Yapay sinir ağının görevi ise modelin
en iyi çıktı skorunu vereceği W ağırlığı ve b bias parametrelerinin hesabını yapmaktır. Sinir ağları sadece doğrusal ağırlıklı toplam (Wx+b) işlemini yaptğında çıktılar sınırlı kalmaktadır. Sinir ağının doğrusal olmayan gerçek hayat problemlerini çözebilmesi için aktivasyon fonksiyonuna sahip olması gerekmektedir. Model, öğrenme işlemini barındırdığı nöronların sahip olduğu ağırlık gibi parametreleri güncelleyerek gerçekleştirmektedir. Böylece derin sinir ağları sınıflandırma, tahmin etme, örnekleme, örüntü tamamlama, nesne 
tespiti, örnekleme ve genelleme gibi problem çeşitlerini çözebilmektedir.

Yaygın olarak kullanılan özelleştirilmiş derin sinir ağlarından bahsetmek gerekirse;
1) Evrişimsel Sinir Ağları CNN (Bilgisayarlı görü uygulamaları)
2) Özyinelemeli Sinir Ağları RNN (Doğal dil işleme, auto translate, sesli asistanlar)
4) Üretici Çekişmeli Ağlar GAN (Sentetik görsel oluşturma)
5) Kapsül Ağları (Capsule Networks)
