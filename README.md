# Aerial-Imagery-Segmentation
Aerial imagery of Dubai obtained by MBRSC satellites (kaggle Dataset) annotated with pixel-wise semantic segmentation in 6 classes.

1- Building: #3C1098

2- Land (unpaved area): #8429F6

3- Road: #6EC1E4

4- Vegetation: #FEDD3A

5- Water: #E2A929

6- Unlabeled: #9B9B9B


# Random Images and its masks from Dataset

![Figure 2021-09-20 142949](https://user-images.githubusercontent.com/31994329/134220917-d0323637-60f0-4627-8626-7b5c5a3a0a43.png)

# Predictions

the colors of pred mask is different from true mask bacause i have assigned each class to a specfic integer from 0 to 5 (6 classes) and didn't convert it back to its RGB values

![Predictions](https://user-images.githubusercontent.com/31994329/134220947-c3e634c3-0530-4068-a01f-05b7dc95f591.png)


# Model Accuracy

![Aerial_Imagery_Model Accuracy](https://user-images.githubusercontent.com/31994329/134220810-42d01dc6-2316-43ae-a323-df95c2014e75.png)

# Model Loss

![Aerial_Imagery_Model Loss](https://user-images.githubusercontent.com/31994329/134220770-e88711dd-b355-4e99-adf2-69d948901d77.png)

# Model Performance per Epoch (usng CSVLogger Callback)

[train_performance_per_epoch.csv](https://github.com/Ahmed-Fayed/Aerial-Imagery-Segmentation/files/7205192/train_performance_per_epoch.csv)
