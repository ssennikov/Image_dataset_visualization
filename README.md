# image_dataset_visualization
This project is about visualizing image dataset in 3-d and 2-d dimensions using embeddings. The dataset contains 1318 .png images of cats representing different breeds
and consequently colors. Initial dataset was unlabeled. We labeled 800 images with 6 classes depending on cats color and breed, then we used 500 images for training 
the model and 300 for its validation. The model used is ResNet18 with pre initialized wheights. 

For visualization we used embeddings extracted from penultimate layer (avgpool) and run them through three demensionality reduction algorhithms: PCA, t-SNE and UMAP.
The final visualization will be looked similar to this:

![image](https://user-images.githubusercontent.com/102593339/198358168-b1c23abb-8db9-4b94-b031-668a3b30a881.png) ![image](https://user-images.githubusercontent.com/102593339/198358496-72cc1c6d-0c4d-45ab-a37c-7c5ea031d63f.png)


To run the code just open the main.py in your IDE. Preliminarily you should install all the libraries from "requirements.txt".
Link to the labelled dataset on Google Drive: https://drive.google.com/file/d/1R4m2WfzepctE841qMVgyCwU_YdQFkKUy/view?usp=sharing
