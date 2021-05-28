# Forschungsarbeit: continual unsupervised learning

## 1. Introduction and Motivation
Continual learning is an important ability for humans and animals. With the ability they can continually acconmmodate new kmowledge while still remember previously learned knowledge. It exhibits positive transfer between old experiences and new knowledge in the same domain or across different domains. The main difficulty in continual learning is caled "catastrophic forgetting", which easily leads to an abrupt performance decrease. 

To overcome this problem, lots of approches have been implemented. Usually they can be divided into four groups:

- **Rehearsal** is a simple mehtod that directly replays the old data.
- **Generative Replay** aims to learn how to generate old data. Compared with rehearsal method, it learns the data distribution and generates pseudo old data by a generative model such as Deep Generative Replay.
- **Dynamic Architectures** are addaptive to a new task by changing the stucture such as [Progressive Neural Network](https://arxiv.org/abs/1606.04671) , [PathNet](https://arxiv.org/abs/1701.08734).
- **Regularization Techniques** have an additional regularization term that can protect important parameters for old tasks in the training such as [Elastic Weight Consolidation (EWC)](https://www.pnas.org/content/pnas/114/13/3521.full.pdf) , [Synapic Intelligence (SI)](http://proceedings.mlr.press/v70/zenke17a/zenke17a.pdf)



Machine learning has two main domains, namely supervised learning and unsupervised learning. Compared supervised learning, unsupervised learning is more applicable because labels are usually expensive to obtain and lots of unlabeled data is available.

If a system has both abilities of continual learning and unsupervised learning, it is very powerful because of real-time learning from dynamic data without labels and quick consolidation and transfer of knowledge. However, continual unsupervised learning attracts little attention because it is very difficult and complex. Up to now, almost all advanced continual learning studies are based on supervised learning, no further experiments on unsupervised learning. 

My research thesis is an starting point in continual unsupervised learning. The main tasks include implementations of two SW architectures (basic Autoencoder and  [Adversarial Autoencoder (AAE)](https://arxiv.org/abs/1511.05644)), experimental verification of the problem "catastrophic forgetting" in continual unsupervised learning, research on solutions that have been proposed and successfully applied to continual supervised learning and implementations of two methods that can mitigate catastrophic forgetting in continual unsupervised learning.



## 2. basics

### 2. 1 AE-$l_2$-Rehearsal

**>>>  [l2 normalized Autoencoder](https://ieeexplore.ieee.org/document/8489068) with rehearsal method**



![AE](/Users/yun/PycharmProjects/FA/photos/AE.png)

