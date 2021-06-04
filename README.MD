# 1. REFERENCES
1. WANG, Xiangwen, et al. You are your photographs: Detecting multiple identities of vendors in the darknet marketplaces. In: Proceedings of the 2018 on Asia Conference on Computer and Communications Security. 2018. p. 431-442.
2. DONG, Chao, et al. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 2015, 38.2: 295-307.
3. Resnet18 <https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html>
4. Super Resolution Using CNN <https://github.com/yjn870/SRCNN-pytorch>

# 2. CONCEPT
Drug images on the Darknetmarket(DNM) are poor quality by default and contain text such as watermarks or drug specifications placed by vendors.  
Therefore, we thought that if we improve the image quality using Super-Resolution and use them for machine learning, the machine learning model will be improved and the civil detection will be better.

# 3. DATASET
We used the DNM data used in the previous paper, but because the entire data was too large, we only used the 3-days data from the Silk Road2.  

* Entire data <https://archive.org/download/dnmarchives/silkroad2.tar.xz>
* SilkRoad2 for 3-days
<https://drive.google.com/file/d/1lAyT54pkWKEVtEIr3MUIKwUFbFPgDCFt/view?usp=sharing>   
_Download this, unzip it and place it in your folder with the name 'DATASET_sample'._

The data in DNM format was converted into an image format that we can check. Then, Super-Resolution was applied to the collected images. In here, we just use a pre-trained SRCNN model(use srcnn_x2.pth).  
_Run the below command in the same folder where you place 'DATASET_sample'. This will create an 'image_collection' folder with about 20,000 images. and 'image_collection_sr' folder with results of Super-Resolution._

    python dnm_to_image.py
    python image_SR.py

<img src="">  
<img src="">

Then we split the two image sets into a test set and a train set that have usernames as labels so that we use them for machine learning.  
_Run the below command in the same folder. This will create an 'user_collection' folder and 'user_collection_sr' folder._

    python distribute_image.py
    python distribute_image_SR.py

# 4. MACHINE LEARNING
We tested the original dataset and Super-Resolution dataset using the RESNET18 model, respectively, and saved the results in the 'runs' folder. we can check the results with TensorBoard.  
_Run the below command in the same folder. This will create an 'runs' folder that has results of civil detection._

    python Resnet18.py
    python Resnet18_SR.py
    tensorboard --logdir=runs --host localhost

If tensorboard doesn't work, try this

    del /q %TMP%\.tensorboard-info\*
    tensorboard --logdir=runs --host localhost

# 5. RESULTS
The results were different from our expectations. We expected that the civil detection results would be improved when training with the Super-Resolution applied dataset, but the results of the two datasets were the same. To confirm the experiment, Accuaration and Loss were compared during the test. The results are mainly analyzed in two ways.
1. The first is that the behavior of Super-Resolution itself can remove the characteristics of the user. Because this is wrong about the concept itself, it seems that a little more analysis is needed.
2. Second, Analyzing the Super-Resolution datasets, the image area of ​​the data was judged to be improved, but the text area seemed to be worse in some way. We expect that these good and bad effects may not change the results.  
* *So, we are planning an experiment to check the difference in Super-Resolution application between the image area and the text area.*
