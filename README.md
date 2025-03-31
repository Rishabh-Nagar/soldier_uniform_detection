# Soldier Uniform Detection

## Project Overview
This project aims to identify soldiers based on their uniforms using deep learning. The trained model classifies soldiers belonging to one of the following forces:
- **CRPF (Central Reserve Police Force)**
- **BSF (Border Security Force)**
- **Jammu & Kashmir Police**

The model is trained using **YOLOv8** and handles edge cases to improve robustness.

---

## Model Training
The model was trained on a dataset containing images of soldiers in different uniforms. The training process was completed in **168 epochs** using YOLOv8n. Below are the key details:


### **Model Performance**

| Class      | Images | Instances | Box(P) | R     | mAP50 | mAP50-95 |
|-----------|--------|-----------|--------|-------|-------|----------|
| All       | 30     | 135       | 0.944  | 0.922 | 0.966 | 0.764    |
| BSF       | 10     | 46        | 0.923  | 0.913 | 0.962 | 0.707    |
| CRPF      | 9      | 39        | 0.948  | 0.897 | 0.981 | 0.776    |
| JK Police | 11     | 50        | 0.960  | 0.955 | 0.957 | 0.808    |

---

## **Edge Case Handling**
To improve real-world robustness, the model was augmented with edge-case scenarios, such as:
- **Low-light conditions** (using brightness contrast adjustments)
- **Occlusions** (simulated with Cutout augmentations)
- **Motion blur** (to simulate moving soldiers)
- **Partial visibility** (cropping images to mimic occlusions)

### **Code for Edge Case Augmentation**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),  # Simulates low-light conditions
        A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.5),  # Simulates occlusion
        A.MotionBlur(p=0.3),  # Simulates motion blur
        A.RandomSizedCrop((180, 224), 224, 224, p=0.4),  # Partial visibility
        ToTensorV2()
    ])
    
    augmented = transform(image=image)
    return augmented['image']
```

## **Conclusion**
This project successfully classifies soldiers' uniforms using a trained YOLOv8 model. The model has been optimized for real-world challenges through augmentation techniques and edge case handling. 🚀

---