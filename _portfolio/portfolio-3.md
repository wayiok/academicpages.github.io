---
title: "ISIC - Detect Skin Cancer (Work in Progress)"
excerpt: "This project is part of the ISIC 2024 Challenge, which aims to improve early skin cancer detection through artificial intelligence. It focuses on building a model to classify skin lesions as malignant or benign using smartphone-quality images and 3D Total Body Photography data. The goal is to enhance diagnostic accuracy, support clinical workflows, and prioritize high-risk cases to ultimately reduce mortality rates. Currently, the work is in the exploratory data analysis phase, laying the groundwork for developing an effective detection algorithm. <br/><img src='https://wayiok.github.io/academicpages.github.io/images/portfolio/p3.png'>"
collection: portfolio
---

# Introduction

To advance the field of automated skin cancer detection, the International Skin Imaging Collaboration (ISIC) has launched this competition. This challenge aims to:

- Improve accuracy in distinguishing between malignant and benign lesions
- Enhance efficiency in clinical workflows
- Develop algorithms capable of prioritizing high-risk lesions
- Ultimately reduce mortality rates associated with skin cancer through early detection

The competition provides smartphone-quality images of skins lesions. From these images the goal is to detect the probability of skin cancer.

The ISIC 2024 Challenge focuses on:

- Binary classification of skin lesions (malignant vs. benign/intermediate)
- Utilization of 3D Total Body Photography (TBP) derived images
- Integration of patient metadata for improved diagnosis
- Prioritization of lesions for clinical review

3D Total Body Photography (TBP) plays a pivotal role in this challenge, offering comprehensive imaging of a patient's entire skin surface. This technology is crucial for:

- Detecting new lesions
- Monitoring changes in existing lesions
- Providing context for individual lesion assessment
- Enabling efficient full-body skin examinations

# Background 

The three major types of skin cancer are:

- Basal Cell Carcinoma (BCC)
- Squamous Cell Carcinoma (SCC)
- Melanoma

BCC and SCC are very common, with over 5 million estimated cases in the US each year, but relatively unlikely to be lethal. The Skin Cancer Foundation estimates that melanoma, the deadliest form of skin cancer, will be diagnosed over 200,000 times in the US in 2024 and that almost 9,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.

![Body 3D](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-1.png)

# Overview 

The goal of this competition is to detect skin cancer using smartphone-quality images of skin lesions. Triaging applications have a significant potential to benefit underserved populations and improve early skin cancer detection, the key factor in long-term patient outcomes.

## Dataset Analysis

The dataset – the SLICE-3D dataset, containing skin lesion image crops extracted from 3D TBP for skin cancer detection – consists of diagnostically labelled images with additional metadata. The following are examples from the training set. 'Strongly-labelled tiles' are those whose labels were derived through histopathology assessment. 'Weak-labelled tiles' are those who were not biopsied and were considered 'benign' by a doctor.

![Tiles](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-2.png)

Dataset contains the following colums variables. 

| **Field Name** (train only) | **Description** | **Field Name** (both) | **Description** |
|-----------------------------|-----------------|-----------------------|-----------------|
| target | Binary class {0: benign, 1: malignant}. | isic_id | Unique case identifier. |
| lesion_id | Unique lesion identifier. Present in lesions that were manually tagged as a lesion of interest. | patient_id | Unique patient identifier. |
| iddx_full | Fully classified lesion diagnosis. | age_approx | Approximate age of patient at time of imaging. |
| iddx_1 | First level lesion diagnosis. | sex | Sex of the person. |
| iddx_2 | Second level lesion diagnosis. | anatom_site_general | Location of the lesion on the patient's body. |
| iddx_3 | Third level lesion diagnosis. | clin_size_long_diam_mm | Maximum diameter of the lesion (mm).+ |
| iddx_4 | Fourth level lesion diagnosis. | image_type | Structured field of the ISIC Archive for image type. |
| iddx_5 | Fifth level lesion diagnosis. | tbp_tile_type | Lighting modality of the 3D TBP source image. |
| mel_mitotic_index | Mitotic index of invasive malignant melanomas. | tbp_lv_A | A inside lesion.+ |
| mel_thick_mm | Thickness in depth of melanoma invasion. | tbp_lv_Aex | A outside lesion.+ |
| tbp_lv_dnn_lesion_confidence | Lesion confidence score (0–100 scale).+ | tbp_lv_B | B inside lesion.+ |
|  |  | tbp_lv_Bext | B outside lesion.+ |
|  |  | tbp_lv_C | Chroma inside lesion.+ |
|  |  | tbp_lv_Cext | Chroma outside lesion.+ |
|  |  | tbp_lv_H | Hue inside the lesion; calculated as the angle of A\* and B\* in LAB\* color space. Typical values range from 25 (red) to 75 (brown).+ |
|  |  | tbp_lv_Hext | Hue outside lesion.+ |
|  |  | tbp_lv_L | L inside lesion.+ |
|  |  | tbp_lv_Lext | L outside lesion.+ |
|  |  | tbp_lv_areaMM2 | Area of lesion (mm²).+ |
|  |  | tbp_lv_area_perim_ratio | Border jaggedness... Values range 0–10.+ |
|  |  | tbp_lv_color_std_mean | Color irregularity... |
|  |  | tbp_lv_deltaA | Average A contrast (inside vs. outside lesion).+ |
|  |  | tbp_lv_deltaB | Average B contrast (inside vs. outside lesion).+ |
|  |  | tbp_lv_deltaL | Average L contrast (inside vs. outside lesion).+ |
|  |  | tbp_lv_deltaLBnorm | Contrast between the lesion and its immediate surrounding skin... Typical values range from 5.5 to 25.+ |
|  |  | tbp_lv_eccentricity | Eccentricity.+ |
|  |  | tbp_lv_location | Classification of anatomical location...+ |
|  |  | tbp_lv_location_simple | Classification of anatomical location, simple.+ |
|  |  | tbp_lv_minorAxisMM | Smallest lesion diameter (mm).+ |
|  |  | tbp_lv_nevi_confidence | Nevus confidence score (0–100 scale)...+ ,++ |
|  |  | tbp_lv_norm_border | Border irregularity (0–10 scale)...+ |
|  |  | tbp_lv_norm_color | Color variation (0–10 scale)...+ |
|  |  | tbp_lv_perimeterMM | Perimeter of lesion (mm).+ |
|  |  | tbp_lv_radial_color_std_max | Color asymmetry... Values range 0–10.+ |
|  |  | tbp_lv_stdL | Standard deviation of L inside lesion.+ |
|  |  | tbp_lv_stdLExt | Standard deviation of L outside lesion.+ |
|  |  | tbp_lv_symm_2axis | Border asymmetry... Values range 0–10.+ |
|  |  | tbp_lv_symm_2axis_angle | Lesion border asymmetry angle.+ |
|  |  | tbp_lv_x | X-coordinate of the lesion on 3D TBP.+ |
|  |  | tbp_lv_y | Y-coordinate of the lesion on 3D TBP.+ |
|  |  | tbp_lv_z | Z-coordinate of the lesion on 3D TBP.+ |
|  |  | attribution | Image attribution, synonymous with image source. |
|  |  | copyright_license | Copyright license. |

In the following bar plots we can see the amount of bening classified skin tiles have a higher number to the malignant ones. This is a unbalanced difference that we will need to change. 

![Bar1](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-3.png)

![Bar2](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-4.png)

Also we can see that mayor malignant lesions are found in the posterior and anterior torso.  

![Bar3](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-5.png)

Age distribution suggests that trainning data is taken mainly from adults in age range from 35 to 80.
![Bar3](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-6.png)

## Target Column

The target column is a binary classifier that we can use to have relevant rows to use as starting point, and with the use of area, perimeter and ratio we can see in a graphical way the tiles that are being used. 

![TilesRatio](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-7.png)
![TilesRatio](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-8.png)
![TilesRatio](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-9.png)

![ScatterPlot](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-11.png)


## Evaluation 
Target colum is defined by a probability of the image tile being bening or malignant, making this a regression problem. Submissions are evaluated on partial area under the ROC curve (pAUC) above 80% true positive rate (TPR) for binary classification of malignant examples.
The receiver operating characteristic (ROC) curve illustrates the diagnostic ability of a given binary classifier system as its discrimination threshold is varied. The shaded regions in the following example represents the pAUC of two arbitrary algorithms (Ca and Cb) at an arbitrary minimum TPR:

![TilesRatio](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-10.png)

# Training Image Classification Model 

As a baseline we use a pre-defined convolutional neural network architecture called EfficientNet. The baseline only takes into account the labeled images and still is not using the available metadata. 

![ScatterPlot](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-12.png)

![ScatterPlot](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-13.png)

![ScatterPlot](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-14.png)

Results of inference trained model.

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-15.png)
![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-16.png)
![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-17.png)

Confusion Matrix for top label classification. 

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p3-18.png)
