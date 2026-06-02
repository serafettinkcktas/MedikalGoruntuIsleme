import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, render_template, request, url_for
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cihaz Seçimi (GPU varsa CUDA, yoksa CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
        self.down1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = double_conv(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        u2 = torch.cat([u2, c2], dim=1)
        c3 = self.conv_up2(u2)
        
        u1 = self.up1(c3)
        u1 = torch.cat([u1, c1], dim=1)
        c4 = self.conv_up1(u1)
        
        return self.final_conv(c4)


CLASS_NAMES = {0: 'Glioma', 1: 'Meningioma', 2: 'Notumor', 3: 'Pituitary'}

def load_models():
    """Modelleri bilgisayardaki .pth dosyalarından yükler"""
    global model_unet, model_classifier
    
    model_unet = UNet(in_channels=3, out_channels=1).to(device)
    unet_path = 'unet7447basari.pth' 
    if os.path.exists(unet_path):
        model_unet.load_state_dict(torch.load(unet_path, map_location=device))
        print("U-Net modeli başarıyla yüklendi. ✅")
    else:
        print(f"UYARI: {unet_path} bulunamadı! Lütfen weights dosyasını belirtilen klasöre koyun.")
    model_unet.eval()

    base_model = models.efficientnet_b0(weights=None)
    in_features = base_model.classifier[1].in_features 
    base_model.classifier = nn.Sequential(
        nn.Linear(in_features, 256), 
        nn.ReLU(),
        nn.Dropout(0.4),              
        nn.Linear(256, 4) 
    )
    model_classifier = base_model.to(device)
    
    classifier_path = 'models/efficientnet_9413finetuned.pth'
    if os.path.exists(classifier_path):
        model_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print("EfficientNet sınıflandırma modeli başarıyla yüklendi. ✅")
    else:
        print(f"UYARI: {classifier_path} bulunamadı! Lütfen weights dosyasını belirtilen klasöre koyun.")
    model_classifier.eval()

load_models()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_and_segment(image_path):
    """Görüntüyü kademeli olarak işler ve sonuç raporu döner"""
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = img_transform(img_pil).unsqueeze(0).to(device)
    
    report = {}
    processed_img_url = None
 
    with torch.no_grad():
        mask_output = model_unet(img_tensor)
        mask_np = torch.sigmoid(mask_output).cpu().squeeze().numpy()

    binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
    pixel_count = np.sum(binary_mask == 255)
    
    if pixel_count > 10:
        
        with torch.no_grad():
            class_output = model_classifier(img_tensor)
            _, predicted_idx = torch.max(class_output, 1)
            pred_class = CLASS_NAMES[predicted_idx.item()]
            
        if pred_class.lower() == 'notumor':
            report['status'] = "TÜMÖR BULUNAMADI (Sağlıklı)"
            report['description'] = "U-Net modeli bölgesel bir şüphe algılamış olsa da, EfficientNet sınıflandırma modeli dokunun 'Sağlıklı' (Notumor) olduğuna karar vermiştir."
            processed_img_url = image_path
            
        else:
            report['status'] = "TÜMÖR TESPİT EDİLDİ"
            report['pixel_area'] = int(pixel_count)
            report['type'] = pred_class
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    loc_x = "Sağ" if cX > 112 else "Sol"
                    loc_y = "Üst" if cY < 112 else "Alt"
                    report['location'] = f"{loc_x} {loc_y} Bölge (Merkez: X:{cX}, Y:{cY})"
                else:
                    report['location'] = "Merkezi Bölge"
            else:
                report['location'] = "Tespit Edilemedi"
                
            report['description'] = f"U-Net modeli kesit üzerinde {pixel_count} piksellik lezyon alanı saptadı. EfficientNet-B0 modeli, dokuyu '{pred_class}' olarak sınıflandırdı."

            orig_img = cv2.imread(image_path)
            h, w, _ = orig_img.shape
            
            mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            color_mask = np.zeros_like(orig_img)
            color_mask[mask_resized == 255] = [0, 0, 255]
            
            overlay_img = cv2.addWeighted(orig_img, 0.7, color_mask, 0.3, 0)
            
            contours_resized, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours_resized, -1, (0, 255, 0), 2) # Yeşil kontur
            
            filename, ext = os.path.splitext(os.path.basename(image_path))
            processed_filename = f"{filename}_processed{ext}"
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, overlay_img)
            
            processed_img_url = f"static/uploads/{processed_filename}"
        
    else:
        report['status'] = "TÜMÖR BULUNAMADI (Sağlıklı)"
        report['description'] = "Yapılan U-Net segmentasyon analizinde doku üzerinde kritik bir lezyon veya tümör formasyonuna rastlanmamıştır."
        processed_img_url = None
        
    return report, processed_img_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Dosya yükleme alanında hata oluştu.", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Herhangi bir dosya seçilmedi.", 400
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            report, processed_img = predict_and_segment(filepath)
            
            return render_template(
                'index.html', 
                uploaded_img=file.filename, 
                processed_img=processed_img, 
                report=report
            )
            
    return render_template('index.html', uploaded_img=None, processed_img=None, report=None)

if __name__ == '__main__':
    app.run(debug=True)