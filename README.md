# SUGPool

A PyTorch implementation of **Structure-based Updatable Graph Pooling for Graph Classification**.

![image](https://github.com/SourceCode1037/SUGPool/blob/main/image.png)

## Abstract

Graph classification is a basic graph analytics tool and has various applications such as molecular function prediction. 
Existing top-k selection graph pooling methods mainly focus on measuring each node more accurately by proposing more complex measurement methods. Specifically, these complex measurement methods will lead to an increase in time complexity and eliminate the effect of speed improvement caused by graph pooling dimension reduction. Besides, during the graph pooling, important nodes will be sampled according to a certain strategy and the information of the unsampled nodes will be lost at the same time. Obviously, it is extraordinarily irrational to lose a massive amount of information of unsampled nodes. 
In this paper, we propose a novel graph classification method called Structure-based Updatable Graph Pooling (SUGPool). 
SUGPool takes into account both the graph structure and the features of nodes in the pooling process, which maximizes utilization of graph information theoretically. In addition, in order to reserve the information of unsampled nodes, we propose a novel strategy for updating these nodes. Experimental results show that, in seven benchmark datasets, our proposed model has an average improvement of 4.3\% comparing with other state-of-the-art methods, and especially achieves the best performance on six datasets, which is a considerable improvement.

## Requirements
- python == 3.6.10
- torch == 1.5.1
- torch_geometric == 1.5.0
- torch_sparse == 0.6.7

All required libraries are listed in [requirements.txt](https://github.com/SourceCode1037/SUGPool/blob/main/requirements.txt) and can be installed with
```python 
pip install -r requirements.txt
```
### Datasets

A collection of benchmark datasets for graph classification and regression is publicly available at [here](https://chrsmrrs.github.io/datasets/). You can change the dataset by modifing the ```args.dataset``` in the following code statement, the program will automatically create a folder called ```data``` to store the datasets.

```python 
dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
```
### Run
  To run SUGPool, execute the following command to train and score on the default dataset.
```python 
python main.py
```

[comment]: <> (## Cite)

## Licence

The code is released under the [Apache-2.0 License](https://github.com/SourceCode1037/SUGPool/blob/main/LICENSE). 
