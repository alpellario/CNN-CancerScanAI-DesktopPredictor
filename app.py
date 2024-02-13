# yapay zeka işlemleri için gerekli kütüphaneler
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# PyQt5 kütüphanesinden gerekli modülleri içe aktarıyoruz.
# QtWidgets, grafiksel kullanıcı arayüzü (GUI) öğelerini kullanmamızı sağlar.
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QDesktopWidget


# 'breastCancerDetectionUI' modülünden 'Ui_MainWindow' sınıfını içe aktarıyoruz.
# Bu sınıf, Qt Designer ile oluşturulan arayüzün Python kodunu içerir.
from breastCancerDetectionUI import Ui_MainWindow


import subprocess
import os
import random
from PyQt5.QtGui import QPixmap

random.seed(100)

# 'App' sınıfı, ana pencereyi oluşturmak için 'QMainWindow' ve 'Ui_MainWindow' sınıflarını genişletir.
class App(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        # 'super()' fonksiyonu, 'QMainWindow' sınıfının kurucu fonksiyonunu çağırarak,
        # bu sınıfın tüm özelliklerini ve metodlarını 'App' sınıfına aktarır.
        super(App, self).__init__()
        # 'setupUi' metodu, Qt Designer ile oluşturulan arayüzü bu pencereye yükler.
        self.setupUi(self)


        # 'tabStart' , "tabDataset" ve "tabMetrics" sekmeleri açılabilir olarak ayarlanıyor "tabPrediction" ise kapalı geliyor, sadece bir image seçildiğinde aktif oluyor.
        self.tabStart.setEnabled(True)
        self.tabDataset.setEnabled(True)
        self.tabPrediction.setEnabled(False)
        self.tabMetrics.setEnabled(True)

        # tabStart'ı başlangıçta seçili olarak ayarla
        self.tabWidget.setCurrentWidget(self.tabStart)

        # 'widget''in arka plan rengini değiştiriyoruz.
        self.widget.setStyleSheet("background-color: #ADD8E6; border-radius: 5px;")
        # #234678 #ADD8E6 #000080

        self.lb_img_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #e7e7e7;")


        # Model yüklendi yazısını başlangıçta gizliyoruz.
        self.lb_model_success.setVisible(False)

        # 'Devam Et' düğmesini başlangıçta etkisizleştiriyoruz (False).
        self.btn_next_1.setEnabled(False)

        # Model yükle butonunun tıklanma olayına import_model() metodunu bağlıyoruz.
        self.btn_model_upload.clicked.connect(self.import_model)

        # Notebook'u aç butonunun tıklanma olayına open_notebook() metodunu bağlıyoruz.
        self.btn_open_notebook.clicked.connect(self.open_notebook)

        # ilk sayfadaki ileri butonunun tıklama olayına switch_to_tabDataset() medotunu bağlıyoruz.
        self.btn_next_1.clicked.connect(self.switch_to_tabDataset)

        # Tahmin tabındaki "yeni image seç" butonunun olayına switch_to_tabDataset() medotunu bağlıyoruz.
        self.btn_new_image.clicked.connect(self.switch_to_tabDataset)

        # Resimlerin yolu ve sayfa bilgileri
        self.image_folder = "prepare_dataset_final\\test"  # Resimlerin bulunduğu dizin
        self.images_per_page = 20
        self.current_page = 0
        self.total_pages = 0
        self.image_paths = []

        # Butonların tıklanma olatlarını bağlıyoruz.
        self.btn_gallery_next.clicked.connect(self.next_page)
        self.btn_gallery_back.clicked.connect(self.previous_page)
        
        # Resimleri yükle
        self.load_images()

        # Seçilen label'ı saklamak için
        self.selected_label = None  

        # Train ve Test datasetlerini yükleme butonlarının olaylarını bağlama
        self.btn_switch_train.clicked.connect(self.switch_to_train)
        self.btn_switch_test.clicked.connect(self.switch_to_test)

        # Veriseti çok büyük olduğundan train ve test datasetleri arası geçişlerde program kasıp hata verebildiği için geçiş butonları devredışı bırakılmıştır.
        self.btn_switch_train.setEnabled(False)
        self.btn_switch_test.setEnabled(False)

        # 36 tane random resmi tahmin eden butonunun tıklama olayını bağlama
        self.btn_random_predict.clicked.connect(self.group_predict_image)

        # Tahmin butonunun tıklama olayını bağlama
        self.btn_img_predict.clicked.connect(self.predict_image)

        # Modeli yükle
        self.model = self.load_model()  # Modeli yükle

        # Tahmin sonrası gözüken resmin etrafındaki border'ı başlangıçta beyaz yap
        self.p_border.setStyleSheet("background-color: white; ")

        # Metrikler ve grafikleri yükleyen fonksiyonu başlangıçta çağır
        self.load_metrics()


        # Model grafiklerinin path'lerini tutan listeyi tanımla
        self.image_paths_2 = [
            "images\model_loss.png",
            "images\model_acc.png",
            "images\conf_matrix.png"
        ]
        # Grafikler combobox'ının onchange olayını bağla
        self.cb_graphics.currentIndexChanged.connect(self.on_combobox_changed)

        # Uygulama açıldığında ilk grafik gösterilsin
        QtCore.QTimer.singleShot(0, lambda: self.cb_graphics.setCurrentIndex(1))

        
    # Modeli seçme Fonksiyonu
    def import_model(self):
        # QFileDialog nesnesi oluşturarak kullanıcıya dosya seçtiriyoruz.
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Model File (*.h5)")  # Sadece .h5 uzantılı dosyaları listele
        file_dialog.setWindowTitle("Choose Model File")    # Diyalog penceresinin başlığı
        file_dialog.setFileMode(QFileDialog.ExistingFile) # Yalnızca mevcut dosyaları+ seçilebilir yap

        # Diyalog penceresi açılır ve kullanıcı bir dosya seçerse işlemleri gerçekleştir
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()  # Seçilen dosyaların yollarını al
            if file_paths:  # Eğer dosya seçildiyse
                file_path = file_paths[0]  # İlk seçilen dosyanın yolunu al
                self.tb_model_path.setText(file_path)  # Dosya yolunu metin kutusuna yazdır

                # Modelin başarıyla seçildiğini belirten etiketi görünür yap
                self.lb_model_success.setVisible(True)

                # 'Devam Et' düğmesini etkinleştir (True) eğer dosya seçildiyse
                self.btn_next_1.setEnabled(bool(file_path))

   
    #Seçilen modeli yükleme fonksiyonu
    def load_model(self):
         # Model dosyasının yolunu alın ve modeli yükleyin
        model_path = self.tb_model_path.text()  # Bu satırı, model dosyasının yolunu doğru bir şekilde alacak şekilde düzenleyin
        if model_path:  # Yol boş değilse modeli yükle
            model = tf.keras.models.load_model(model_path)
            return model
        else:  # Model yolu boşsa bir uyarı göster
            print("Model file path is not specified.")
            return None
        
    # Google Colab'da eğitilen modelin notebook'unu Visual Studio Code ile açar
    def open_notebook(self):
        #Python Notebook'unun dosya adını ver.
        filepath = "Breast-Cancer-Model-Training.ipynb"

        if filepath:
            try:
                # Visual Studio Code'u subprocess ile başlat
                subprocess.Popen(["code", filepath], shell=True)
            except Exception as e:
                print(f"Hata: {e}")



    # İlk sayfada modeli seçtikten sonra devam et butonuna tıkladığında çalışan fonksiyon
    def switch_to_tabDataset(self):
        # tabWidget içindeki tabDataset tabına geçiş yap
        self.tabWidget.setCurrentIndex(1)  # Buradaki 1, tabDataset'in index numarasıdır



    # Test veristeni tabDataset'e yükleme
    def load_images(self):
        # test verisetindeki .png ile biten tüm image pathlerini image_path'se ver
        self.image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith('.png')]
        # resim sayısını bir sayfada gösterilecek (20) resim sayısına bölüp kaç sayfa image olduğunu belirle
        self.total_pages = len(self.image_paths) // self.images_per_page
        
        # İmage dizisini karıştır
        random.shuffle(self.image_paths)
        # Resimleri galeriye yükleme
        self.update_gallery()



    # Yüklenen resimleri image_paths'den alıp galeride gösteren fonksiyon
    # Program ilk yüklendiğinde, test ve train datasetleri arasında geçiş yapıldığında çalışır
    def update_gallery(self):
        # Mevcut sayfada gösterilecek resimleri hesaplama
        start = self.current_page * self.images_per_page
        end = start + self.images_per_page
        page_images = self.image_paths[start:end]

        # Resimleri etiketlere yerleştirme ve tıklama olaylarını ayarlama
        for i, img_path in enumerate(page_images):
            label = getattr(self, f'p{i+1}')
            pixmap = QPixmap(img_path)
            scaled_pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)
            
            # Varsayılan stil uygulama
            label.setStyleSheet("border: 4px solid #e7e7e7; border-radius: 5px;")
                        
            # Tıklama olayını ayarlama
            label.mousePressEvent = lambda event, label=label, path=img_path: self.on_image_click(event, label, path)

            # Çift tıklama olayını ayarlama ve Tahmin etme fonksiyonunu çağırma
            # label.mouseDoubleClickEvent = lambda event: self.predict_image()
            
            label.mouseDoubleClickEvent = lambda event, path=img_path: self.predict_image()

        # Sayfa numaralarını güncelleme
        self.lb_current_page.setText(str(self.current_page + 1))
        self.lb_total_page.setText(str(self.total_pages + 1))

    # Galeride ilerideki sayfalara ilerleme 
    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.update_gallery()

    # Galeride önceki sayfalara gitme
    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_gallery()

    # Galeriye train datasetini yükleme
    def switch_to_train(self):
        # Eğitim veri setine geçiş yap
        self.image_folder = "prepare_dataset_final\\train"  # Eğitim veri setinin bulunduğu dizin
        self.current_page = 0  # Sayfa numarasını sıfırla
        self.load_images()     # Resimleri yeniden yükle


    # Galeriye test datasetini yükleme
    def switch_to_test(self):
        # Test veri setine geçiş yap
        self.image_folder = "prepare_dataset_final\\test"   # Test veri setinin bulunduğu dizin
        self.current_page = 0  # Sayfa numarasını sıfırla
        self.load_images()     # Resimleri yeniden yükle


    # Galerideki herhangi bir image'e tıklandığında çalışan fonksiyon
    # Still ekleme ve resme ait classı değiştirme gibi işlemler yapıyor
    def on_image_click(self, event, label, img_path):
        self.lb_img_label.setStyleSheet("color: #e7e7e7; font-size: 28px;")
        # Önceki seçilen resmin sınırını ve gölge efektini sıfırla
        if self.selected_label is not None:
            self.selected_label.setStyleSheet("border: 4px solid #e7e7e7; border-radius: 5px;")
            self.selected_label.setGraphicsEffect(None)

         # Seçilen resmin dosya yolunu sakla
        self.selected_img_path = img_path

        # Resmin ismini ve kanser durumunu güncelleme
        img_name = os.path.basename(img_path)
        # Resmin isminin son 20 karakterini al ve göster
        self.lb_img_name.setText(img_name[-20:])
        # Sondan 5. karakterinden classı al ve resme tıkladığında Kanserli veya Kansersiz olduğunu göster
        img_class = img_name[-5]
        self.lb_img_label.setText("Cancerous Cell" if img_class == '1' else "Non-Cancerous Cell")

        # classın 1 veya 0 olma durumlarına göre renklendirme ve stillerini tanımla
        if img_class == '1':
            label.setStyleSheet("border: 4px solid red; border-radius: 5px;")
            self.lb_img_label.setStyleSheet("color: red; font-size: 28px;")
        else:
            label.setStyleSheet("border: 4px solid green; border-radius: 5px;")
            self.lb_img_label.setStyleSheet("color: green; font-size: 28px;")
        
        # Yeni seçilen resmin sınırını ve gölge efektini uygula
        self.apply_shadow_effect(label)

        # Seçilen resmi saklıyoruz
        self.selected_label = label
 

    #Kendisine parametre olarak verilen image path'i ni modele vermeye hazır hale getiren fonksiyon
    def prepare_image(self, img_path):
        # Resmi cv2 ile oku ve BGR'dan RGB'ye çevir
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resmi yeniden boyutlandır
        img_resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
        
        # Resmi bir NumPy dizisine dönüştür ve boyutlandır
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, 0)  # Modelin beklediği şekle getir (batch boyutunu ekle)
    
        return img_array


    # Galerideki resimlerin üzerine çift tıklandığında veya Tahmin et butonun tıkladığında tetiklenen fonksiyon
    # Her seferinde tek bir resmi tahmin etmek için kullanılır
    def predict_image(self):
        model = self.load_model()
        if self.selected_img_path and model:  # Eğer resim seçildiyse ve model yüklendiyse
            self.tabWidget.setCurrentIndex(2)  # Tahmin sekmesine geçiş yap
            self.tabPrediction.setEnabled(True) # Tahmin sekmesine enable yap

            #Tahmin sekmesine seçilen resmi yükleme ve boyutlandırma
            pixmap = QPixmap(self.selected_img_path)
            self.p_image.setPixmap(pixmap.scaled(self.p_image.width(), self.p_image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            #Seçilen resmi modele vermeye hazır hale getirme (boyutlandirma, numpya çevirme vs)
            prepared_image = self.prepare_image(self.selected_img_path)
            #Hazırlanmış resmi modele verip tahmin ettirme
            prediction = model.predict(prepared_image)
            #Tahmin edilen clası alma ve confidence(güven aralığı) nı belirleme
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            self.lb_true.setText(self.lb_img_label.text())  # Gerçek sınıfı güncelle
            self.lb_predict.setText("Cancerous Cell" if predicted_class == 1 else "Non-Cancerous Cell")  # Tahmin sonucunu güncelle
            self.lb_confidence.setText(f"{confidence * 100:.2f}%")  # Güven aralığını güncelle
            self.lb_confidence.setStyleSheet('font-size: 28px; font-weight: bold;')

            
            # Image'in gerçek değerinin yazdığı label'ı renklendirme
            # Eğer seçilen image'in gerçek sonucu kanserli hücre ise kırmızı, değilse yeşil renk uygula
            if self.lb_true.text() == "Cancerous Cell":
                self.lb_true.setStyleSheet("color: red; font-size: 28px; font-weight: bold;")
            else:
                self.lb_true.setStyleSheet("color: green; font-size: 28px; font-weight: bold;")

            # Image'in tahmin değerinin yazdığı label'ı renklendirme
            # Eğer tahmin sonucu kanserli hücre ise kırmızı, değilse yeşil renk uygula
            if self.lb_predict.text() == "Cancerous Cell":
                self.lb_predict.setStyleSheet("color: red; font-size: 28px; font-weight: bold;")
            else:
                self.lb_predict.setStyleSheet("color: green; font-size: 28px; font-weight: bold;")


            
            # Tahmin sonucunu kontrol et ve image'in etrafındaki border'ı yeşil veya kırmızı yap
            if self.lb_true.text() == self.lb_predict.text():                
                print("Prediction is correct")
                self.p_border.setStyleSheet("background-color: green; ")
                
            else:
                print("Prediction is wrong")
                self.p_border.setStyleSheet("background-color: red; ")
        
        else:
            print("No image selected or model could not be loaded.")
            #alert oluştur ve pencerenin merkezinde göster
            alert = QtWidgets.QMessageBox()
            alert.setText("No image selected or model could not be loaded.")
            alert.exec_()
            alert.show()


    # Galerinden her classa ait toplamda 36 tane rastgele image'i tahmin edip yeni pencerede gösteren fonksiyon
    def group_predict_image(self):
        print("36 random pictures are predicted...")
        #Eğitilen modeli yükleme fonksiyonu çalıştır
        model = self.load_model()
        if model: #Eğer model var ise
            #Random seed'i sıfırla
            #Başlangıçta galeri hep aynı olsun diye seed 100 verilmişti
            random.seed()
            
            # Her image'in adının sondan 5.hanesi hangi classa ait olduğunu gösteriyor
            # Eğer 1 ise kanserli imagelere ekle
            # Eğer 0 ise kansersiz imagelere ekle
            tumor_images = [path for path in self.image_paths if path[-5] == '1']
            healthy_images = [path for path in self.image_paths if path[-5] == '0']
            # Kanserli ve kansersiz imagelerden 18 er tane rastgele image seç
            selected_tumor_images = np.random.choice(tumor_images, 18, replace=False)
            selected_healthy_images = np.random.choice(healthy_images, 18, replace=False)

            # Seçilen resimleri hazırlama ve birleştirme
            images = []
            image_paths  = []
            for img_path in np.concatenate([selected_tumor_images, selected_healthy_images]):
                prepared_img = self.prepare_image(img_path)
                images.append(prepared_img)
                image_paths .append(img_path)

            # Tahmin sonuçlarını göstermek için yeni bir pencere oluştur
            dialog = QDialog(self)
            dialog.setWindowTitle("Predictions Results")
            dialog.setLayout(QtWidgets.QVBoxLayout())
            canvas = FigureCanvas(plt.Figure(figsize=(10, 10)))
            dialog.layout().addWidget(canvas)

            ax = canvas.figure.subplots(6, 6)

            # Her resmi ayrı ayrı tahmin et ve sonuçları göster
            for i, img_path in enumerate(image_paths):
                # Resmi hazırla ve tahmin et
                prepared_image = self.prepare_image(img_path)
                prediction = model.predict(prepared_image)
                print(prediction)
                #Tahmin değerini 0 a veya 1 e yuvarlayarak hangi classa ait olduğunu belirle
                predicted_class_idx = prediction.argmax()
                print(predicted_class_idx)
                true_class_idx = 1 if img_path[-5] == '1' else 0
                # Tahmin sayfasındaki Gerçek Label ve Tahmin edilen Label değerlerini gir
                predicted_label = 'Cancerous' if predicted_class_idx == 1 else 'Non-Cancerous'
                true_label = 'Cancerous' if true_class_idx == 1 else 'Non-Cancerous'

                # Görselleştirme
                ax[i // 6, i % 6].imshow(prepared_image[0] / 255.0)
                title_color = 'red' if predicted_class_idx != true_class_idx else 'black'
                ax[i // 6, i % 6].set_title(f'Actual: {true_label}\nPred: {predicted_label}', color=title_color)
                ax[i // 6, i % 6].axis('off')


            canvas.figure.subplots_adjust(top=0.95, bottom=0.05, hspace=0.6, wspace=0.4)

            # Ekranın çözünürlüğünü alın
            screen = QApplication.desktop().screenGeometry()

            # Pencerenin boyutunu ve konumunu ayarlayın
            dialog.setGeometry(0, 0, screen.width(), screen.height())

            # Pencereyi göster
            dialog.exec_()
        else: #eğer model seçilmedi ise hata verdir
            print("No image selected or model could not be loaded.")
            #alert oluştur ve pencerenin merkezinde göster
            alert = QtWidgets.QMessageBox()
            alert.setText("No image selected or model could not be loaded.")
            alert.exec_()
            alert.show()
        
        



     # Grafikleri yükleyen fonksiyon
    def load_metrics(self):
        print("Charts are loading...")

    #Grafik combobox'ı değiştinde tetiklenen fonksiyon
    def on_combobox_changed(self, index):
        # Seçilen model grafiğine göre görseli p_graph label'ına yerleştir
        pixmap = QtGui.QPixmap(self.image_paths_2[index])
        self.p_graph.setPixmap(pixmap.scaled(self.p_graph.size(), 
                                             Qt.KeepAspectRatio, 
                                             Qt.SmoothTransformation))


    # Geleride üzerinde bir kere tıklanan resme gölge ve classına göre (kirmizi. yesil) border ekleme
    def apply_shadow_effect(self, widget):
        # Image'lere gölge efekti oluşturma
        # sol alta dogru biraz daha şeffaf ve smooth bir spread etkisine sahip ios tarzı gölge oluşturma
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(45)  # Gölge bulanıklık yarıçapını artırarak gölgeyi daha dağılmış yap
        shadow.setXOffset(3)      # Yatay ofseti azaltarak gölgenin daha yakın olmasını sağla
        shadow.setYOffset(3)      # Dikey ofseti azalt
        
        # Gölge rengini ayarlama
        if(self.lb_img_label.text() == "Cancerous Cell"):
            # Kanserli Hücre ise kırmızı gölge yap
            shadow.setColor(QColor(255, 0, 0, 240))
        else:
            # Kansersiz Hücre ise yeşil gölge yap
            shadow.setColor(QColor(0, 255, 0, 240))
    
        # Gölge efektini widget'a uygulama
        widget.setGraphicsEffect(shadow)




# Python dosyası doğrudan çalıştırıldığında aşağıdaki kod bloğu çalışır.
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # 'App' sınıfından bir örnek oluşturuluyor ve 'mainWin' değişkenine atanıyor.
    mainWin = App()
    # Oluşturulan pencere görünür hale getiriliyor.
    mainWin.show()
    # Program, pencere kapatılana kadar çalışmaya devam eder.
    sys.exit(app.exec_())
