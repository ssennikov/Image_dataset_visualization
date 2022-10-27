# Cats clusterization via hidden representation 
This project is about visualizing image dataset in 3-d and 2-d dimensions using 512-d hidden representation. The dataset contains 1318 .png images of cats representing different breeds and colors. Initial dataset was unlabeled, so we labeled 800 images with 6 classes depending on cats color and breed, then we used 500 images for training the model and 300 for its validation. After that, using ResNet18 with pre initialized wheights, we built classificator to label the rest of the data.

Here you can see the metrics of the classification model:

![image](https://user-images.githubusercontent.com/102593339/198383042-1f3b9a2a-4b4c-44bb-898b-4581d4e42717.png)


For visualization we used embeddings extracted from avgpool layer of the model and run them through three demensionality reduction algorhithms: PCA, t-SNE and UMAP.
The final visualization will be looked similar to this:

![image](https://user-images.githubusercontent.com/102593339/198381403-8ef53a60-87dc-48a0-b67b-33bd47240520.png) ![image](https://user-images.githubusercontent.com/102593339/198381486-419c122c-2e6c-428a-ab95-c769fbcf6719.png) ![image](https://user-images.githubusercontent.com/102593339/198382496-c5f4ab20-021d-4528-a127-128a8c4712c6.png)


To run the code open the main.py.
Link to the labelled dataset on Google Drive: https://drive.google.com/file/d/1R4m2WfzepctE841qMVgyCwU_YdQFkKUy/view?usp=sharing
