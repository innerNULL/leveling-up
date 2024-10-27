# Paper Note -- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)

## Keywords
**transfer learning**, **adapter**, **definition of adapter**

## Overview
This paper proposes a new parameter effective yet simple **transfer learning** 
approach.
   
Before this paper, there were two typical transfer leaning approachs:
* **fine-tuning**: Continue training base model/pre-training model/foundation
 model on down-stream task's dataset. Sometime to make base model suit in down 
 stream tasks, you may need at extra layers at the end of base model. During 
the transfer leaning process, all parameters of base model will be tuned.
* **feature-based transfer**: Using base model as a "feature extractor", 
which means we only using base model the extract certain representation 
of the input, on typical example is embedding. Similar with **fine-tuning**, 
this method usually add extra layer(s) at the end of base model to suit it 
in down stream tasks, but the difference is during the transfer learning 
process the parameters of base model will be frozen.

This paper raised a new approach called **adapter basee tuning** or **adapter**. 
This approach is similar with **feature-based transfer**, as during transfer 
learning the parameters of based model will be frozen. The main difference is 
instead of only adding extrac layer(s) at the end of based model, the **adapter 
 based tuning** integrate **adapter** in several internal layers of base model.
Here **adapter** can just be any simple neural network architectures which can be 
added together with certain internal layer's output of base model.

