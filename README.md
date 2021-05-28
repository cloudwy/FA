# Forschungsarbeit: continual unsupervised learning



## 1. Introduction and Motivation
Continual learning is an important ability for humans and animals. With the ability they can continually acconmmodate new kmowledge while still remember previously learned knowledge. It exhibits positive transfer between old experiences and new knowledge in the same domain or across different domains. The main difficulty in continual learning is caled "catastrophic forgetting", which easily leads to an abrupt performance decrease. 

To overcome this problem, lots of approches have been implemented. Usually they can be divided into four groups:

- **Rehearsal** is a simple mehtod that directly replays the old data.
- **Generative Replay** aims to learn how to generate old data. Compared with rehearsal method, it learns the data distribution and generates pseudo old data by a generative model such as [Deep Generative Replay (DGR)](https://arxiv.org/pdf/1705.08690.pdf).
- **Dynamic Architectures** are addaptive to a new task by changing the stucture such as [Progressive Neural Network](https://arxiv.org/abs/1606.04671) , [PathNet](https://arxiv.org/abs/1701.08734).
- **Regularization Techniques** have an additional regularization term that can protect important parameters for old tasks in the training such as [Elastic Weight Consolidation (EWC)](https://www.pnas.org/content/pnas/114/13/3521.full.pdf) , [Synapic Intelligence (SI)](http://proceedings.mlr.press/v70/zenke17a/zenke17a.pdf)



Machine learning has two main domains, namely supervised learning and unsupervised learning. Compared supervised learning, unsupervised learning is more applicable because labels are usually expensive to obtain and lots of unlabeled data is available.

If a system has both abilities of continual learning and unsupervised learning, it is very powerful because of real-time learning from dynamic data without labels and quick consolidation and transfer of knowledge. However, continual unsupervised learning attracts little attention because it is very difficult and complex. Up to now, almost all advanced continual learning studies are based on supervised learning, no further experiments on unsupervised learning. 

My research thesis is an starting point in continual unsupervised learning. The main tasks include implementations of two SW architectures (basic Autoencoder and  [Adversarial Autoencoder (AAE)](https://arxiv.org/abs/1511.05644)), experimental verification of the problem "catastrophic forgetting" in continual unsupervised learning, research on solutions that have been proposed and successfully applied to continual supervised learning and implementations of two methods that can mitigate catastrophic forgetting in continual unsupervised learning.



## 2. basics

### 2. 1 AE-l2-Rehearsal

**>>>  [l2 normalized Autoencoder](https://ieeexplore.ieee.org/document/8489068) with rehearsal method**

![image](https://github.com/cloudwy/FA/blob/master/photos/AE.png)

The model is a simple model and it can be divided into three parts:

1. The main body is l2 normalized autoencoder. Compared with a basic autoencoder, it has an additional layer on the output of the encoder. L2 normalization has two advantages: make latent code more separate and improve the accuracy with k-means.
2. Clustering with k-means.
3. Rehearsal Technique to overcome catastrophic forgetting. The static decoder has the same structure as the decoder in l2 normalized autoencoder. It can directly generate old data via latent code.

### 2.2 AAE-DGR

![image](https://github.com/cloudwy/FA/blob/master/photos/AAE.png)

The model is a more complex model and it can be divided into two parts:

1. The main body is an unsupervised AAE, which is implemented based on the Semi-Supervised AAE in the above mentioned paper.
2. Deep Generate Replay (DGR) technique. In the paper, a scholar includes a generator and a solver, but here only has a generator. The part is implemented a static decoder, whose input is the samples from Gaussian distribution and Categorical distribution and output is generated images.



## 3. Experiment Settings

Plattform: Python 3.6 and TensorFlow 1.12

Dataset: Split-MNIST and Split-FashionMNIST

Incremental class learning: Split the dataset into 5 tasks, each with 2 non-overlapping classes

Train the model with current task, but test the model with not only current task but also all old tasks

Replayed dataset: based on fixed memory ($repl\_batch\_size = batch\_size - \lfloor\frac{batch\_size}{task}\rfloor$)

Three steps for each model:

- **Lower Bound**: Training sequentially ***without any strategies*** to verify whether catastrophic forgetting exists.
- **With Strategies**: Training sequentially ***with a strategy*** to overcome catastrophic forgetting.
- **Upper Bound**: Training sequentially ***with all old real data*** to evaluate how well the strategy works.



## 4. Results and Visualization

**AE-l2-Rehearsal**

![image](https://github.com/cloudwy/FA/blob/master/photos/Result-AE.png)

The results with rehearsal technique are almost overlapped. In the Error bars, small variance illustrates a stable model.

![image](https://github.com/cloudwy/FA/blob/master/photos/umap-AE.png)

[Uniform Manifold Approximation and Projection(UMAP)](https://arxiv.org/abs/1802.03426) is a technique for dimension reduction and visualization. With this technique the pictures are plotted. Here is an example of the AE-l2-Rehearsal model based on Fashion-MNIST dataset. The left picture is plotted with original features and labels, then the picture in the middle is with latent features and cluster labels and the right picture is with latent code and predicted label, which is evaluated from cluster labels and ground truth.

The clusters in the right picture are more widely distributed compared with the results in the first picture.

**AAE-DGR**

![image](https://github.com/cloudwy/FA/blob/master/photos/Result-AAE.png)

From the results, DGR technique seems to work and helps to reduce catastrophic forgetting from task1 to task4, but it is completely helpless in the task5. The specific reason hasn't been found. The possible reasons are as follows: 

- AAE is unstable, sometimes its performance is suddenly decreased, but after several iterations it can be  recovered.
- DGR works, but it makes AAE more unstable based on several experiments.

![image](https://github.com/cloudwy/FA/blob/master/photos/umap-AAE.png)

There are many connections between two clusters, which leads to bad performance.



## 5. Future Works

1. Improve the AAE model.
2. Choose other powerful techniques to overcome catastrophic forgetting, for example, change parameters instead of replaying data.



## 6. Code

My repository includes the following files and folders:

Files:

- models: contains all implemented models.
- datasets: used to generats a dataset.
- utils: contains utility functions such as initialize training data, plot function and so on.

Folders:

- **AAE**: experimental files related to AAE-DGR model
- **AE**: experimental files related to AE-l2-Rehearsal model

- **Ref_paper**: related papers during the period of research thesis.
- **photos**: photes about results and visualization
- **Error bar**: coding files to plot the error bars.
- **explore_code**: some discarded files

