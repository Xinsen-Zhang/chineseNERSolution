# chineseNERSolution
中文 NER 解决方案尝试

## 数据

数据为《CLUENER 细粒度命名实体识别》, 来自https://github.com/CLUEbenchmark/CLUENER2020

>  由于这是用来学习的数据, 模型的开发, 仅使用 train.json 和 dev.json 即可

## run docker with pytorch
sudo docker run -p 23:22 --name="torch-remote" -v /workspace/remote_work --restart=always -itd --shm-size 42g --gpu all nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

## 训练 word2vec
python train_word2vec.py --tag=char --size=300
python train_word2vec.py --tag=char --size=250
python train_word2vec.py --tag=char --size=200

## changelog
20201229 学习 lstm 的压缩 + spatialDropout 两项技术