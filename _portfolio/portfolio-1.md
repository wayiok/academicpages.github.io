---
title: "Cryo-electron Tomography  Object Identification"
excerpt: "Short description of portfolio item number 1<br/><img src='/images/500x300.png'>"
collection: portfolio
---

The "CZII - CryoET Object Identification" competition on Kaggle challenges participants to identify small biological structures within large 3D volumes obtained through cryo-electron tomography (Cryo-ET). This report shows the solution implemented with a 3D-Unet and 2D object detection model for multi-class segmentation, the results obtained show that both approaches achieve accuracy in more than 50\

# Introduction

The CryoET Object Identification challenge is funded by the
Chan Zuckerberg Initiative and its primary objective is to acquire more knowledge about protein complexes for cellular function, which are essential for disease treatments. The available data obtained from tomographs are often available in a standardized format, and the analysis of this specific information is challenging when identifying the types of protein complexes within these images. 

Cryo-electron tomography opens the door to the study of the structure of unique objects, such as cell structures and even entire cells \cite{stewart2017cryo}. To do this, multiple images of the sample are taken at different inclinations within the microscope (generally from -70º to 70º), which are subsequently processed using specialized programs to reconstruct its three-dimensional structure, as seen in the figure. The available dataset provided in the competition contains already classified and denoised images of tomographs, the classification includes six particle types with varying prediction difficulty: apo-ferritin (easy), beta-amylase (not scored, impossible), beta-galactosidase (hard), ribosome (easy), thyroglobulin (hard), and virus-like-particle (easy), with beta-amylase excluded from scoring due to its evaluation challenges.

![Cryo-electron tomography (cryoET)](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-ap1.png)

# Methodology

## Dataset

The dataset consists of 7 cryo-electron tomography (cryoET) images, represented as 3D tomograms where each voxel corresponds to a 10x10x10 nm cube, as seen in the Figure. Each tomogram contains various objects of interest, whose locations are provided as centroid coordinates in associated files. Objects include ribosomes, virus-like particles, apo-ferritin, thyroglobulin, and B-galactosidase, with radius ranging from 6 to 15 voxels. The challenge allows a voxel-level labeling to be considered correct if it falls within half the particle's radius from the actual centroid. There are associated files to each tomogram containing x, y, z coordinates of object centroids.

![Voxel Segmentation](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-ap2.png)

Synthetic data has been used to train models to detect these objects. This data is generated with realistic characteristics mimicking the tomograms, serving as a proxy for real-world samples, especially when annotated real tomograms are limited.

For the second architecture implemented, the preparation of the datasets for training converts 3D volumetric data into 2D images slices, this reduces memory requirements and address data scarcity. Key steps  in this process include normalizing the data, creating image slices, generating YOLO-compatible annotations, and organizing datasets into structured folders for training and validation.

## Architectures Implemented

There are numerous architectures, methods and approaches that have proven to be especially effective in certain object detection tasks and for extracting features of tomograms. Among these, the YOLO (You Only Look Once) network \cite{diwan2023object} and 3D U-NET \cite{agrawal2022segmentation} stand out.

### 3D U-NET

3D U-Net is a convolutional neural network (CNN) architecture designed specifically for image segmentation tasks, where the goal is to classify each pixel (or voxel in 3D cases) in the input image. It is particularly well-suited for biomedical image analysis, making it ideal for the cryo-electron tomography (cryoET) dataset.\\
Key features of the model 3D U-Net are:

* Encoder-Decoder Structure:

  * Encoder: responsible for capturing contextual information by downsampling the input image through convolutional, max-pooling layers and extracting high-level features.
  * Decoder: responsible for reconstructing the spatial details by upsampling the features back to the input resolution and producing a dense segmentation map.

* Skip Connections: while encoding, the model also send the outputs to the corresponding layers and the Decoder help recover fine-grained spatial details lost during downsampling. These connections concatenate feature maps from the Encoder with those in the Decoder, enhancing localization accuracy.
* 3D Adaptation: For the CryoET dataset, the 2D U-Net is extended to a 3D version, where 3D convolutions and pooling operations are applied, enabling the model to process volumetric data and segment objects in 3D space effectively.

### YOLO

This method incorporates a real-time object detection stage, which uses a convolutional neural network to divide a 2D image into regions and predict the coordinates and probabilities of existence of the objects in each region. YOLO has the advantage of being fast, accurate and robust against different lighting conditions, size and shape of the objects.

Several studies have applied YOLO in detection tasks from medical images: such as mammography  [Al-Masni et al., 2018](https://www.nature.com/articles/s41592-018-0259-9), the study of melanoma [Nie et al., 2019](https://ieeexplore.ieee.org/document/8970033) and dental diseases [Sonavane & Kohar, 2022](https://link.springer.com/chapter/10.1007/978-981-16-6285-0_12), achieving accuracies and sensitivities above 90% in laboratory simulations and studies with patients, for internal validation data. These results may be an indicator of the effectiveness of the method in detecting abnormal objects in biomedical tasks, as well as its potential to become a novel approach, capable of performing disease detection and classification with good performance in clinical routine, which could also have relevant implications for this specific competition.

The Python programming language and PyTorch machine learning frameworks were used. Additionally, tools and libraries such as YOLOv5 from Ultralytics were employed to facilitate model training and object detection. The YOLO architecture was implemented using a pre-trained model (YOLO11), which incorporates recent advancements in object detection and data augmentation. Data pre-processing and augmentation included techniques like rotation, shear, flipping, and mix-up during training. The development and training process used a machine equipped with an NVIDIA L4 GPU (22.5 GB GDDR6), providing the computational power necessary to efficiently train the model + 235 GB HDD storage.

The latest YOLO11 model architecture, as shown in following figure, is composed by three main parts:

![YOLO11 Pre-Trained Model Architecture](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-ap3.png)

* Backbone: Is the deep learning architecture that acts as a feature extractor.
* Neck: Combines the features acquired from the various layers of the backbone model.
* Head: Predicts the classes and bounding box regions which is the final output produced by the object detection model. 

## Training Process

### Transfer Learning with Synthetic Data

Before training the model with real-world data, we opted for a transfer learning approach by pre-training the model on synthetic data. Synthetic data often provides a controlled environment where particle features, distributions, and annotations are more reliable and consistent than in real data. Pre-training allows the model to learn general features and patterns that are transferable to real-world data, such as recognizing particle shapes and boundaries.

### Pre-trained 3D U-NET

During training, the validation metric used in this model is Dice Metric, which is commonly used in segmentation tasks to evaluate the overlap between the predicted segmentation and the ground truth. It is not the same as "accuracy" in a traditional classification sense but is instead a measure of how well the predicted and true segmentation align.

The Dice score ranges from 0 to 1:

* 1 indicates perfect overlap (the prediction is exactly the same as the ground truth).
* 0 indicates no overlap.

It is computed as: 

$$
Dice~Score = \frac{2|A \cap B|}{|A| + |B|}
$$

Where:

* A is the predicted segmentation
* B is the ground truth segmentation.

![Validation Score Performance during training phase](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-1.png)

The model performs well while training with the validation score increasing.\\  And as for the loss function used in 3D U-Net model is the Tversky Loss, which is particularly suited for imbalanced segmentation tasks, especially when one class significantly dominates the others.

The Tversky Loss is a generalization of the Dice Loss and is defined as:

$$
\mathcal{L}_{\text{Tversky}} = 1 - \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}
$$

Where:
- \( TP \) = True Positives
- \( FP \) = False Positives
- \( FN \) = False Negatives
- \( $\alpha, $\beta \) are weights controlling the penalty for FP and FN respectively.

In this equation if $\alpha = $\beta = 0.5, this loss is equivalent to the Dice Loss.

![Loss Performance during training phase](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-2.png)

The model's loss initially decreases well at first but then struggles to improves from around epoch 50.

### YOLO 

The training configuration includes the following parameters and optimization techniques: 

* Epochs: The model is trained for 100 full passes through the dataset.
* Warm-up Epochs: Gradual increase in the learning rate over the first 10 epochs.
* Batch Size: The number of samples processed in one training step is 32.
* Image Size: Input images are resized to 640 x 640 pixels.
* Data Augmentation: 
  * Rotation: Up to ±45 degrees.
  * Shear: Up to 5 degrees.
  * Horizontal and Vertical Flipping: Probabilities of 0.5 for both.
  * Mixup: A data augmentation technique that blends two training images.
  * Copy-Paste: Augmentation by combining regions from different images.
* Optimizer: AdamW
* Seed: 8620 – Used for reproducibility.
* Initial Learning Rate: 0.0003

Losses in the following figure are divided in three terms that define the overall loss function used in object detection models like YOLO11. The total loss function for object detection is given by:

$$
\text{Loss} = \alpha \cdot \text{DFL Loss} + \beta \cdot \text{CLS Loss} + \gamma \cdot \text{BOX Loss}
$$

Where:
* $\alpha : Weight for the Distribution Focal Loss (DFL Loss),
* $\betha : Weight for the Classification Loss (CLS Loss),
* $\gamma : Weight for the Bounding Box Loss (BOX Loss).

The total loss can be calculated programmatically as:

$$
\text{Total Loss} = \alpha \cdot dfl + \beta \cdot cls + \gamma \cdot box
$$

![YOLO Training Losses Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-3.png)

# Experimental evaluation setup

## 3D U-Net

During evaluation phase on the test dataset, we follow the official metric score of the Kaggle competition which focuses on precision, recall, and F-beta score.

1. **Precision**: Measures the proportion of correctly predicted objects (hits) among all predicted objects.  
   $$
   \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
   $$

2. **Recall**: Measures the proportion of correctly predicted objects among all ground-truth objects.  
   $$
   \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
   $$

3. **F-beta Score**: A weighted harmonic mean of precision and recall, emphasizing recall (with \( \beta = 4 \)):  
   $$
   F_{\beta} = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}
   $$
   In this case, \( \beta = 4 \) means recall is weighted 16 times more than precision.

In the context of this challenge, a particle is considered a "true" positive if it lies within 0.5 times the particle's radius of the ground truth particle, this tolerance helps account for some variability in particle locations, allowing small shifts while still counting as a correct prediction. And the particles are divided into 3 types and weighted differently:

* Easy Particles (ribosome, virus-like particles, apo-ferritin) are assigned a weight of 1.
* Hard Particles (thyroglobulin and $\beta$-galactosidase) are assigned a weight of 2. This weighting scheme reflects the relative difficulty of detecting each particle type, with harder particles having more influence on the final score. The hard particles are particularly prioritized, making recall critical for them.
* Impossible Particles: Beta-amylase particles are included in the training data but do not contribute to the score, as they have a weight of 0 in the scoring mechanism. Even if predicted, they do not affect the final evaluation.
\end{enumerate}

The final F-beta score is computed by summing the per-particle scores, applying the weights to each particle type, and normalizing by the total weight:

$$
\text{Final\_lb\_score} = \sum \left( \frac{F_{4}(\text{particle}) \cdot \text{weight}(\text{particle})}{\text{weight}(\text{particle})} \right)
$$

This gives a Final lb\_score that reflects the model's ability to identify particles correctly, with emphasis on the harder particles and recall.

## YOLO

The following figure  graph shows the evaluation metrics on the YOLO11 model over 100 training epochs. Each line tendency and definition are described below:

- **Precision (Blue Line):** Measures how many detected objects are classified correctly.  
  The precision starts low and increases rapidly in the first 10 epochs, as the training configuration makes the learning rate increase after 10 epochs. After this, precision stabilizes above **0.6**.

- **Recall (Orange Line):** Measures how many of the ground truth objects are correctly detected.  
  It follows the same increasing trend as precision.

- **mAP50 (Green Line):** Mean Average Precision at 50% IoU (Intersection over Union).  
  This metric shows the accuracy for both classification and localization when the IoU threshold is 50%. In the graph, it increases significantly in the first 10 epochs and then stabilizes around **0.55–0.6**.

- **mAP50-95 (Red Line):** Mean Average Precision across IoU thresholds from 50% to 95%, in increments of 5%.  
  This metric evaluates performance across varying levels of object overlap. It grows more slowly than the other metrics, stabilizing around **0.4–0.45**.

![Evaluation Metric for YOLO Model Training](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-4.png)

This evaluation shows that the model achieves reliable performance in both classification (Precision, Recall) and localization (mAP metrics) after 100 training epochs.

# Results and Discussion

As a result, the 3D-Net model pre-trained with the synthetic data outperforms the baseline model in terms of final F-beta score (0.524 $>$ 0.339). YOLO11 model was trained with the provided competition dataset and outperforms the 3D implementation. Both approaches were published as submission for the kaggle competition.

## Comparison of Architectures

Both architectures implemented, 3D U-NET and YOLO11, are based on convolutional neural networks (CNNs) as their backbone. As their names says, 3D U-NET uses 3D convolutional layers to process volumetric datasets, while YOLO processes object detection in 2D images. 

## Interpretation of results

The following table shows the performance that both models achieved after submission to Kaggle. Submissions are evaluated by calculating the F-beta metric with a beta value of 4. In this case YOLO11 training got the best score, this result can be justified with the fact that YOLO models have faster inference and training time due to their 2D design. The can achieve high accuracy in detecting object centers when working with individual slices. In contrast, 3D U-NET  achieved a 0.33 score but at the cost og higher computational costs. 3D can better capture volumetric context, and with the use of synthetic data we were able to achieve a better score, but we were far behind YOLO11. 

| Model                   | Kaggle Score |
|-------------------------|--------------|
| U-NET Baseline          | 0.339        |
| Pre-trained 3D U-NET    | 0.524        |
| Pre-trained YOLO 11     | 0.625        |

## Object label analysis

### 3D U-Net

From the follwong table, we can observe the following attributes of the model prediction on all the particles:

| **Particle Type**        | **P** | **T** | **Hit** | **Miss** | **FP** | **Precision** | **Recall** | **F-beta=4** | **Weight** |
|--------------------------|-------|-------|---------|----------|--------|----------------|-------------|---------------|-----------|
| apo-ferritin             | 66    | 139   | 51      | 88       | 15     | 0.772727       | 0.366906    | 0.378603      | 1         |
| beta-amylase             | 98    | 31    | 18      | 13       | 80     | 0.183673       | 0.580645    | 0.515152      | 0         |
| beta-galactosidase       | 157   | 40    | 22      | 18       | 135    | 0.140127       | 0.550000    | 0.469260      | 2         |
| ribosome                 | 290   | 142   | 106     | 36       | 184    | 0.365517       | 0.746479    | 0.703357      | 1         |
| thyroglobulin            | 560   | 94    | 83      | 11       | 477    | 0.148214       | 0.882979    | 0.683624      | 2         |
| virus-like-particle      | 96    | 30    | 28      | 2        | 68     | 0.291667       | 0.933333    | 0.826389      | 1         |

1. **"Easy" particles:**
  - Virus-like-particle achieves the highest F-beta score (**0.826**), indicating effective identification.
  - Apo-ferritin had the lowest F-beta score (**0.378**), suffering from low precision.
  - Ribosome also achieves a high F-beta score (**0.703**), second only to Virus-like-particle.

   Overall, the "easy" particles are truly easier to detect — except for Apo-ferritin.

2. **"Hard" particles:**
  - Thyroglobulin, despite being considered harder, shows strong recall, contributing positively to the overall evaluation.
  - Beta-galactosidase suffers from low precision, which reduces its F-beta score.

3. **"Impossible" particle:**  
   Although listed as impossible, the model is able to detect the beta-amylase particle surprisingly well, with an F-beta score of **0.515** — even better than Apo-ferritin and Beta-galactosidase.

The following visualizations Figure 9 illustrate the model’s performance in detecting particles on real-world data TS 6 4.
These visualizations help assess the alignment of the model’s predictions with the ground truth and highlight common
failure cases.

![Visualization of the Detection of the model on data TS 6 4](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-5.png)

We also recognize that particles with high recall often have lower precision, suggesting the model prioritizes identifying true positives but struggles with false positives and "Hard" particles tend to have more challenges, particularly in precision.

### YOLO11

YOLO11 results can be seen in the following prediction matrix, where each cell contains the proportion of predictions for that combination of true and predicted classes. The most important observations are: 

![YOL11 Prediction Matrix](https://wayiok.github.io/academicpages.github.io/images/portfolio/p1-6.png)

* Ribosome: 78\% of the ribosome instances were correctly classified. This is a relatively strong performance for this class.
* apo-ferritin: 62\% of predictions are correct, but there is some misclassification (38\% in other classes).
* virus-like-particle: Performs the best, with 90\% correct predictions.

## Failure Analysis

### 3D U-Net

The model occasionally fails to identify true particles, leading to high recall penalties. In case of the "easy" particle apo-ferritin, the model detected only 51 out of 139 true particles, resulting in a recall of 36.69%. This suggests that the model struggles to generalize apo-ferritin features, possibly due to overlapping characteristics with other particles or insufficient training examples in the synthetic dataset which leads to a high miss rate reducing the F-beta score and the final lb\_score of the model.

### YOLO

For YOLO11 case the particle types beta-amylase and beta galactosidase have significant miss-classification rates, with proportions spread across other classes. For the thyroglobulin class, while 47 of the predictions are correct, it is also misclassified as "ribosome" and "virus-like particle." Also, the background of the images are misclassified, which might indicates a challenge for the model to differentiate background noise from actual particle types.

# Conclusion and Future Work

The research focuses on particle detection and classification in Cryo-ET dataset in which we use 2 CNN-based models YOLO11 and 3D U-Net and as a result, YOLO11 outperforms 3D U-Net in most particle type classifications in terms of F-beta score. It is important to mention that the types of proteins with the highest accuracies are due to the radius around the centroid being the largest in ribosome, thyroglobulin, and virus-like particles.

In the case of YOLO11, we can see that it struggles with objects that span multiple slices; however, 3D U-Net captures volumetric context better, especially for irregularly shaped objects. This can be seen in the improvement of the score after the Kaggle submission baseline. YOLO11 could benefit still from adding of space partitioning data structures to better associate detected objects with their true 3D positions based on centroid data. 

In conclusion, the choice between YOLO11 and 3D U-Net would be in favor of YOLO11 who have proved its potential in object detecting task like this but still should depend on the specific characteristics of the dataset and the task's priorities. Future work could explore hybrid approaches that combine YOLO11's speed and precision with the volumetric insights of 3D U-Net, aiming for a more comprehensive solution to cryo-ET particle detection challenges.
