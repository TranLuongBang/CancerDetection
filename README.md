# Urotherial_Cancer_Cell_Object_Detection
For more information, please check [My Thesis](https://drive.google.com/drive/folders/16RBOZ32QO5u7eiiGsWsOExms0S8cFV2j?usp=sharing)
### Abstract
By leveraging the power of deep neural networks, many problems have started
to be explored based on medical imaging analysis, such as diabetic retinopathy
detection or brain tumor segmentation. However, many other problems remain
underexplored for several reasons, especially the scarcity or non-existence of data.
Therefore, our motivation is to tackle one of these problems, which is detecting
bladder tissue cancer at the cellular level.
In recent years, object detection is growing rapidly from classical detectors based on
Convolutional Neural Networks (CNNs) to detectors based on Vision Transformer.
However, the accuracy of these models in the medical domain is still uncertain. We
want to evaluate the diagnostic accuracy of these detectors in detecting cancer cells.
In this preliminary study, we first focus on collecting the urinary cytology dataset
conducted by Sant’Andrea General Hospitals in Rome and Sapienza University of
Rome. Based on this novel dataset, we analyze and preprocess data. Following this,
we apply state-of-the-art object detection algorithms to train models. Ultimately,
we evaluate the results, which can be used for further decisions.
### Introduction
Bladder cancer is one of the most common cancers in the world. According to the "World Cancer Research Fund International", bladder cancer is the 6th and
17th most common cancer in men and women, respectively, in terms of absolute
incidence. In Italy, bladder cancer is the 5th common cancer after breast, lung,
prostate, and colon cancers. Italy has more than 28,336 new cases of bladder cancer
and 7,108 deaths cases recorded in 2020, which ranked number 3 for the highest rate
of bladder cancer worldwide. The main risk factors for bladder cancers are smoking,
personal or family history of cancer, and exposure to certain chemicals. When
bladder cancer is discovered early on, it is still highly treatable. In 2016, World
Health Organisation (WHO) stated that there are three most common bladder cancer
types: urothelial carcinoma (UC), adenocarcinoma, and squamous cell carcinoma.
Among them, UC is the most prevalent, with about 90% of all bladder cancers.
UC starts in the urothelial cells that line the inside of the bladder. Urothelial
cells also line the ureters, urethra, and renal pelvis. Different tests and diagnostic
technologies such as urinalysis, urine cytology, and medical images like computerized
tomography (CT) urogram or retrograde pyelogram are used to diagnose UC. Urine
cytology is the most widely used diagnostic to detect UC, especially when the UCs
are high-grade. Urine cytology has a high sensitivity of up to 85% in high-grade
UCs but quite low sensitivity for low-grade UCs. However, urine cytology still has
some drawbacks of the need for highly-skilled specialists and low sensitivity. Every
day qualified cytopathologists must analyze a large number of urinary cytology
samples. This procedure is considered time-consuming, complex, and has significant
inter-variability potential. With the success of deep learning in medical imaging
analysis, it could be the critical method for minimizing diagnostic errors and time
requirements. Diagnostic errors can be the incorrect detection between suspected
cancer, cancer, and normal cancer cells. In many applications, the deep learning
model shows high accuracy and sensitivity in diagnosing diseases based on medical
images. It may include urinary cytology images.
### Methodology
***CNNs-based Object Detectors:*** Faster R-CNN, YOLOv5, SSD, RetinaNet 

***Transformer-based Object Detectors:*** Vision Transformer and Swin Transformer
### Dataset
In this study, we use a urinary cytology dataset collected from the Cytopathology
Laboratory Unit hosted in the Sant’Andrea General Hospitals in Rome, collaborating
with the Sapienza University of Rome. This dataset is composed of bladder tissue
images of 81 patients. Good materials are chosen for this project with well-preserved
slides of urinary cytology. All the images were taken by a camera DP27(Olympus)
with an x40 objective lens attached to a microscope BX45 focusing on the interest
area. Dataset criteria include cells in monolayer arrangement and exclude areas with
excess debris, red blood, and inflammation cells.
The dataset is reviewed by an expert cytopathologist and manually identified
categories of cells. To classify urothelial cells, the Paris system criteria is applied.
### Experiment Results
***Performance on different Object Detectors***

![image](https://user-images.githubusercontent.com/18412307/200424487-1e550755-f6ab-45ec-b29b-05e731613a85.png)

***Performance on different Feature Extractors***
![image](https://user-images.githubusercontent.com/18412307/200424506-7cb299c8-411a-44e3-a0a8-7f260f0ed42c.png)

***Performance on different IoU thresholds in Non-Maximum Suppression***
![image](https://user-images.githubusercontent.com/18412307/200424531-7e7089b6-f5bc-446e-869f-99bae617e3a2.png)

***Performance on different Aspect Ratios***
![image](https://user-images.githubusercontent.com/18412307/200424561-a71e9227-40f0-4502-b95f-b09741c2ce56.png)

***Performance on bigger image sizes***
![image](https://user-images.githubusercontent.com/18412307/200426341-2be4032e-4f7f-4c78-a23a-520999dc5a46.png)

***The Best Model***
![image](https://user-images.githubusercontent.com/18412307/200424597-cc4a63c7-bb1a-45b2-8054-84eed43a4c50.png)

### Inference Results
***Ground Truth - Prediction***
![image](https://user-images.githubusercontent.com/18412307/200425413-6df4143e-5fb4-4f9f-a437-34a6285848d2.png)

***Ground Truth - Prediction***

![image](https://user-images.githubusercontent.com/18412307/200425572-eef8557e-14a5-479c-a52c-10f924e74439.png)

### Conclusion

As we mentioned, bladder cancer is one of the most prevalent cancer in the
world, especially in Italy. Diagnosing this cancer need qualified experts, and timeconsuming. As persons working in the data science field, we see the potential of
applying deep learning in detecting this bladder cancer at the cellular level.

In this study, we pushed a lot of effort into data collection and preprocessing.
Choosing the best images of interesting areas and annotating dense cells in highresolution images require time and patience. After all, we provide a novel urinary
cytology dataset. In this dataset, we still find many problems, such as a smallscale dataset, class imbalance, and noise labeling. To overcome these problems, we
have used some statistical methods and deep learning techniques. For example, to
effectively detect dense objects in high-resolution images, we used the technique of
slicing the image into portions of the images with smaller sizes. We trained and
evaluated models on these portions. We performed inference on smaller portions of
the original image before integrating the sliced predictions on the original image.
As a result, we can detect dense and small objects in high-resolution images. By
doing this, along with the data augmentation technique, we generated more data to
train models in the scenario lacking medical images is one of the main problems in
medical image analysis. In addition, the class imbalance is still a big problem when
training the detection model. To tackle this problem, we applied Focal Loss, which
is down the weight of the popular class and up the weight of the rare class. Further
implementation can be generating more cancer cells from existing cancer cells.

Furthermore, we transferred the use of cutting-edge object detectors in general
tasks to the medical domain. There are many object detectors to deal with this
problem. In this study, we showed the performance comparison between CNN-based
and Transformer-based object detectors. While CNNs are inspired by hierarchical
and pyramidal feature maps, Transformers are inspired by multi-head self-attention
increases modeling capacity, which leads to higher accuracies. Experiment results
have shown, Transformer-based detectors outperformed CNN-based detectors making
their potential use in many other computer vision applications. However, object
detection in medical imaging is still a challenging task because of some reasons
shown in Section 1.1.2 and 1.1.3.

To sum up, the accuracy of the detection models on our real-world dataset, where
we find many problems, is not really high but promising. One of the main reasons
could be that due to the huge number of cells in each image, we are not able to
annotate all of them. This leads to the mean precision average is not really high
since models also detect the unannotated cells.

We hope our preliminary study is just the beginning of a more extensive and
comprehensive research undertaking to address the issue of diagnosing cancer cells in urinary cytology images. In addition, more and more in-depth investigations and
efforts in data collection should be made to build a big and reliable dataset for this
problem.
