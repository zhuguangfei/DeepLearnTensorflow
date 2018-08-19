LSTM网络：
    核心思想：引入一个叫做细胞状态的连接，这细胞状态用来存放想要记忆的东西
    (对应于简单RNN中的h,只不过这里不再只存上一次的状态,而是通过网络学习存放那些有用的状态)。
    1、忘记门：决定什么时候需要把以前的状态忘记
        该门决定模型会从细胞状态中丢弃什么信息
        该门会读取h(t-1)和x(t)，输出一个0~1之间的数值给每一个在细胞状态C(t-1)中的数字。1表示完全保留,0表示完全舍弃
        一个语言模型的例子：假设细胞状态会包含当前主语的性别，于是根据这个状态便可以选择正确的代词。当我们看到新的主语时，应该把
        新的主语在记忆中更新。该门的功能就是先去记忆中找到那个旧的主语(并没有真正忘记操作，只是找到而已)
    2、输入门：决定什么时候加入新的状态
        两部分功能：一部分是找到那些需要更新的细胞状态，另一部分是把需要更新的信息更新到细胞状态里。
        忘记门找到了需要忘记的的信息f(t)后，再将它与旧状态相乘，丢弃掉需要丢弃的信息。再将结果加上i(t)*C(t)使细胞状态获得新的信息
        这样就完成了细胞状态的更新
    3、输出门：决定什么时候需要把状态和输入放在一起输出
        在输出门中，通过一个Sigmoid层来确定哪部分的信息将输出，接着把细胞状态通过Tanh进行处理(得到一个在-1~1之间的值)并将它和Sigmoid
        门的输出相乘，得出最终想要输出的那部分，例如在语言模型中，假设已经输入一个代词，便会计算出需要输出一个与动词相关的信息

窥视孔连接(Peephole)：
    是为了弥补忘记门一个缺点:当前cell的状态不能影响到Input Gate，Forget Gate在下一时刻的输出，使整个cell对上个序列的处理丢失了部分信息，
    所以增加了Peephole connections。计算顺序为：
        1、上一时刻从cell输出的数据，随着本次时刻的数据一起输入InputGate和ForgetGate。
        2、将输入门和忘记门的输出数据同时输入cell中。
        3、cell出来的数据输入到当前时刻的OutputGate，也输入到下一时刻的InputGate，ForgetGate。
        4、ForgetGate输出的数据与cell激活后的数据一起作为Block的输出。
    通过这样的结构，将Gate的输入部分增加了一个来源--ForageGate，InputGate的输入来源增加了cell前一时刻的输出，OutputGate的输入来源
    增加cell当前时刻的输出，使cell对序列记忆增强。

带有映射输出的LTMP：
    是在原有LSTM基础上增加一个映射层，并将这个layer连接到LSTM的输入，该映射层是通过全连接网络来实现的，可以通过改变其输出纬度调节
    总的参数量，起到模型压缩的作用

基于梯度剪辑的cell：
    源于这问题：LSTM的损失函数是每一个时间点的RNN的输出和标签的交叉熵之和。这种loss在使用Backpropagation through time(BPTT)梯度
    下降法的训练过程中，可能出现剧烈的抖动。
    当参数值在较为平坦的区域更新时，由于该区域梯度值比较小，此时的学习率一般变得较大，如果突然到达陡峭的区域，梯度值陡增，再与此时较大
    的学习率相乘，参数就有很大幅度的更新，因此学习过程非常不稳定。
    Clipping cell方法的使用可以优化这个问题：为梯度设置阈值，超过阈值的梯度值都会被cut，这样参数更新的幅度就不会过大，因此容易收敛。
    从原理上可以理解为：RNN和LSTM的记忆单元的相关运算是不同的，RNN中每一个时间的记忆单元中的内容(隐藏层节点)都会更新，而LSTM则是使用
    忘记门机制将记忆单元中的值与输入值相加(按某种权重值)再更新(cell状态)，记忆单元中的值会始终对输出值产生影响(除非ForgetGate完全关闭))
    因此梯度值易引起爆炸，所以Clipping功能是很有必要的。

GRU网络：
    功能与LSTM一样，将忘记门和输入门合成一个单一的更新门，同样还混合了细胞状态和隐藏状态及其他一些改动

Bi-RNN网络：
    又叫双向RNN，是采用两个方向的RNN网络
    RNN网络擅长的是对连续数据的处理，既然是连续的数据规律，我们可以学习正向规律还可以学习反向规律，正向和反向结合的网络，会比单向的循环网络有更高的拟合度。
    双向RNN的处理过程与单向RNN非常类似，就是在正向传播的基础上再进行一次反向传播，而且两个连接着一个输出层。这个结构提供输出层输入序列中，每一个点有完整
    的过去和未来的上下文信息。
    双向RNN会比单向RNN多一个隐藏层，6个独特的权值在每一个时步被重复利用，6个权值分别对应输入到向前和向后隐藏层(w1,w3)，隐藏层到隐藏层自己(w2,w5),
    向前和向后隐藏层到输出层(w4,w6)。
    在按照时间序列正向运算完之后，网络又从时间最后一项反向地运算一遍，即把t3时刻的输入与默认值0一起生成反向的out3，把反向out3当成t2时刻的输入与原来的t2时刻
    输入一起生成反向out2；依次类推，直到第一个时序数据。
    双向循环网络的输出是2个，正向一个，反向一个。最终会把输出结果通过concat并联一起，然后交给后面的层处理。

基于神经网络的时序类分类CTC：
    CTC(Connectionist Temporal Classification)是语音辨识中的一个关键技术，通过增加一个额外的Symbol代表NULL来解决叠字问题。在基于连续的时间序列分类任务中，
    常常使用CTC的方法。该方法主要体现在处理loss值上，通过对序列对不上的label添加blank(空label)的方式，将预测的输出值与给定的label值在时间序列上对齐，通过
    交叉熵的算法求出具体损失值。
    ctc_loss函数：
        按照序列来处理输出标签和标准标签之间的损失
        tf.nn.ctc_loss(labels,inputs,sequence_length,preprocess_collapse_repeated=False,ctc_merge_repeated=True,time_magjor=True)
        labels:一个int32类型的稀疏矩阵张量
        inputs:(常用变量logits表示)经过RNN后输出的标签预测值，三维浮点型张量，当time_major为False是形状为[batch_size,max_time,num_class],否则为
                [max_time,batch_size,num_class]
        sequence_length:序列长度
        preprocess_collapse_repeated:是否需要预处理，将重复的label合并成一个label，默认是False
        ctc_merge_repeated:在计算时是否将每一个non_blank(非空)重复的label当成单独的label来解释，默认是true
        time_magjor:决定inputs的格式
        1、preprocess_collapse_repeated=TRUE;ctc_merge_repeated=TRUE
            忽略全部重复标签，只计算不重复标签
        2、preprocess_collapse_repeated=False;ctc_merge_repeated=TRUE
            标准的CTC模式，也是默认模式，不做预处理，只在运算时重复标签将不再当成独立的标签来计算
        3、preprocess_collapse_repeated=TRUE;ctc_merge_repeated=TRUE
            忽略全部重复标签，只计算不重复标签，因为预处理时已经把重复标签去掉了
        4、preprocess_collapse_repeated=TRUE;ctc_merge_repeated=False
            所有重复标签都会参加计算
    inputs中的classes是需要输出多少类，使用ctc_loss时，要将classes+1

SparseTensor(稀疏矩阵)类型：
    SparseTensor(indices,values,dense_shape):
        indices:不为0的位置信息，它是一个二维int64Tensor，shape为(N,ndims)，指定了sparse tensor索引
        values:一个list，存储密集矩阵中不为0位置所对应的值，它与indices里的顺序对应
        dense_shape:一个1D的int64 tensor，代表原来密集矩阵的形状
    生成SparseTensor:
    SparseTensor转dense:
    levenshtein距离:
        又叫编辑距离，是指两个字符串之间，由一个转成另一个所需要的最少编辑操作次数。许可的编辑操作包括：将一个字符替换成另一个字符、插入一个字符、删除一个字符

CTCdecoder:
    ctc_loss中的logits是预测结果，但却是带有空标签，而且是一个与时间序列强对应的输出。需要一个转化好的类似于原始标签的输出。可以使用CTCdecoder，经它处理后
    就可以与标准标签进行损失值的运算了
    CTCdecoder函数：
        tf.nn.ctc_greedy_decoder(inputs,sequence_length,merge_repeated=True)
        tf.nn.ctc_beam_search_decoder(inputs,sequence_length,batch_width=100,top_paths=1,merge_repeated=True)
    解码完的decoder是list，不能直接使用，通常取decoder[0]，然后转成密集矩阵，得到一个批次结果，然后再一条一条地取到每一个样本的结果