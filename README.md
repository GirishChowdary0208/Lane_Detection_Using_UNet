
# LANE DETECTION USING UNET
Lane detection is a crucial computer vision task that involves identifying the boundaries of driving lanes in an image or video of a road scene. It plays a vital role in various applications, particularly in the realm of autonomous vehicles and advanced driver-assistance systems (ADAS).  Convolutional neural networks (CNNs) are now the dominant approach for lane detection. Trained on large datasets of road images, these models can achieve high accuracy even in challenging situations.  In this we implemented UNET architecture which is a deep learning algorithm widely used for image segmentation.
## UNET
U-Net is a powerful, versatile neural network architecture specifically designed for semantic segmentation tasks, which involve dividing an image into different meaningful regions. Lane detection in self-driving cars is a perfect example of a semantic segmentation task, where the goal is to accurately identify and segment the lanes in a road image.  UNET has the ability to extract line features and the ability to extract context improves the accuracy of lane lines. The experimental results show that the improved neural network can obtain good detection performance in complex lane lines, and effectively improve the accuracy and time-sensitives of lane lines.

## TuSimple Dataset
The TuSimple dataset is a large-scale dataset for autonomous driving research, focusing on lane detection and perception tasks. It's widely used in computer vision and autonomous driving communities for benchmarking and developing algorithms.

The TuSimple dataset consists of 6,408 road images on US highways. The resolution of image is 1280×720. The dataset is composed of 3,626 for training, 358 for validation, and 2,782 for testing called the TuSimple test set of which the images are under different weather conditions.



## UNet Architecuture 

![UNet Architecture ](https://github.com/GirishChowdary0208/Lane_Detection_Using_UNet/assets/92716279/43430fdb-ee9b-4db3-8643-e6a1747a4e59)


## Downloads :    
Download the Full Dataset Here: [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple)

Download the PreProcessed Dataset Here: [TuSimple_Preprocessed](https://www.kaggle.com/datasets/rangalamahesh/preprocessed-1/data)

Checkout the Kaggle Link for this project : [Kaggle](https://www.kaggle.com/code/rangalamahesh/lane-detection-using-unet)
## Getting Started 

To run this project you can download the UNET.ipynb file provided in the repository and the dataset from the download section and can implement the whole process by following the instructions in the [Kaggle Link](https://www.kaggle.com/code/rangalamahesh/lane-detection-using-unet).  Below are the basic Requirements to run the code 

```bash
  - Tensorflow version > 2.0.0
  - Keras
  - GPU
  - CUDA
```

I choose Kaggle to implement this because it provides inbuilt GPU accelerator which accelerate the training process, I used GPU T4 x2 to implement this.  You can also choose google colab to run this, google colab also provides inbuilt GPU accelerator which fast up the training process much faster that using CPU.
## Training the Model

To train this model I used GPU T4 x2 accelerator which accelerated my trained process much more faster than using CPU.  In my model training process the training Epochs are 32, batch size is 8 and the process went well with higher accuracy and low loss. 

I used the [TuSimple_Preprocessed](https://www.kaggle.com/datasets/rangalamahesh/preprocessed-1/data) dataset to run the process.  You can prepare you own preprocessed dataset by follwing this [Link](https://www.kaggle.com/code/rangalamahesh/preprocessed).
You have to download or upload the [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) to prepare your own dataset.



### Test 

You can download the weights file Lane_Model_2.h5 file and directly test it for predictions.  

Also find the inference.ipynb file which contained the testing or inference code.

To test the code
```bash
  Download the inference_unet.ipynb file and load the model weights
  Lane_Model_2.h5 path  and provide the testing image path in the inference code. 
  By running the inference_unet.ipynb file you can visualize the plot of the predictions.
```

## METRICS VISUALIZATION

![UNet Metrics](https://github.com/GirishChowdary0208/Lane_Detection_Using_UNet/assets/92716279/38e39fb6-2b60-4b3e-aa7d-3c1765d705cc)


The Above graph visualize the metrics during the training process, it shows the graph showing Training & Validation Loss and Training & Validation Accuracy with the staring value and ending value.  The graphs shows the gradual decrease in the loss function and gradual increase accuracy as shown in the visualization.

You can also check the TensorBoard logs to visualize the metrics and the layers in the Architecture.

To run the TensorBoard logs follow the command in your Terminal:
```bash
tensorboard --logdir=path/to/your/logs/directory
```
After running the command, open your web browser and go to http://localhost:6006 to access the TensorBoard interface. You'll be able to navigate through the different tabs to explore the data recorded in the tensorboard v2 file.
## Predictions 

![Unet output1](https://github.com/GirishChowdary0208/Lane_Detection_Using_UNet/assets/92716279/c5caeb2d-437a-4acf-b239-46b90b70ce7d)
![Unet output2](https://github.com/GirishChowdary0208/Lane_Detection_Using_UNet/assets/92716279/2dc80427-82b1-4367-8bd3-96978ec7533f)


I tested the Predictions on the inference code by loading the saved.h5 weights file and testing it on the new images.  The model predictions came out to be good as shown in the figures.

## 🔗 Connect with me
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GirishChowdary0208)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/girish-chowdary-919b6522b/)

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/_GirishChowdary)

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/girishchowdary22)

[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Girish-Chowdary)

## 🚀 About Me

I am Girish Chowdary, an enthusiastic and versatile individual deeply passionate about the realm of technology and its endless possibilities.

- 🔭 Embarking on the journey of knowledge at Centurion University of Technology and Management for a BTech in Computer Science and Engineering.

- 🌱 Exploring the vast landscapes of Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, and various Programming Languages to unlock boundless potential.

- 👨‍💻 Showcasing my passion through innovative projects, all neatly organized at https://github.com/GirishChowdary0208.

- 📫 Connect with me via email at mupparjugirishchowdary@gmail.com to share ideas, collaborate, or discuss the limitless possibilities of technology.
