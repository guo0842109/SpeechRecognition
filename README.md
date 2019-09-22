# 语音章节内容
所有涉及语言的问题都比较难以理解，这方面可以参考我们所讲的文本降维算法。因此对于语音而言，其网络结构通常也更复杂一些。所以在语音识别章节所涉及的内容包括用不同的网络去实现相同的语音识别任务。

学习的内容为： 
- 语音识别：语音->文本

需要理解的部分：
- RNN与CNN
- 序列比对
- 双向RNN

## 需要库函数
Anaconda需要额外的库函数为librosa、tqdm、TensorFLow，librosa主要用于mfcc特征计算，可以利用scipy完整的计算mfcc。
推荐安装方式：
- pip install tqdm
- conda install -c conda-forge librosa

对于pip在安装TensorFLow后可能存在部分问题，解决方式是进行升级：
- conda install -c anaconda html5lib pip

## 关于GPU问题
对于使用GPU用户，存在的主要问题在于显存超出，这个解决方式包括：
- 减少BATCHSIZE
- 将部分计算放置于CPU之中


# 语音数据集
在语音任务之中，困难点在于获取标签数据，这方面有一些开源数据集
- 开源数据集[libri数据](http://www.openslr.org/12/)
- TIMIT数据集此数据集是需要付费，其对于音素的标注比较详细
- 开源中文语音数据库[THCHS30](http://www.openslr.org/18/)
由于中文语音数据比较缺乏，如果今后想进一步进行学习的话，这里提供另外一个解决方式：
**从中文电影中获取电影音轨和字幕**



# 文件说明
## SpeechToTextEN
用于英文语音识别，网络结构预设为双向RNN，下载数据后用datatools/libri_process.py处理下载后的libri数据。处理完成后利用libir_train.py文件进行训练。在数据预处理过程中需要用到flac程序，将flac转换为wav格式。

## SpeechToTextCN
用于中文语音识别，网络结构为ResNet，下载THCHS30用于训练。test.py文件用于测试。预设数据位于data文件夹下

## Speech
用于项目展示，利用SpeechToTextEN之中的模型

## Basic
用于基础学习，主要内容DataFeature.py用于演示mfcc特征生成过程

## Reference
参考文献

## data
用于保存数据

## model
保存了模型数据，模型可能并非是最好的，需要在此基础上再训练
