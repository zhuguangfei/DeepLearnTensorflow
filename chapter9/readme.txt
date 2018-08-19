对于序列化的特征任务，都适合采用RNN网络解决。
有情感分析，关键字提取，语音识别，机器翻译和股票分析等。
1、正向传播
    1、开始时t1通过自己输入权重和0作为输入，生成out1
    2、out1通过自己的权重生成了h1,然后和t2经过输入权重转化后一起作为输入，生成out2
    3、out2通过同样的隐藏层权重生成了h2，然后和t3经过输入权重转化后一起作为输入，生成了out2。
2、随时间反向传播(BackPropagation Through Time,BPTT)
    反向传播BP算法：
        1、有一个批次含有3个数据A、B、C，批次中每一个样本有两个数据(x1,x2)通过权重(w1,w2)来到隐藏层H并生成批次h。
        2、该批次的h通过隐藏层权重p1生成最终的输出结果y。
        3、y与最终的标签p比较，生成输出层less(y,p)。
        4、less(y,p)与生成y的导数相乘，得到Del_y。Del_y为输出层所需要需要的修改值。
        5、将h的转置与del_y相乘得到del_p1。这是源于h与p1相等得到的y。
        6、最终将该批次的del_p1求和并更新到p1上。
        7、同理，再将误差反向传递到上一层：计算Del_h。得到Del_h后再计算del_w1、del_w2并更新。

BasicRNNCell
    def __init__(self,num_units,input_size=None,activation=tanh,reuse=None):num_units:包含cell的个数；input_size:废弃

BasicLSTMCell
    LSTM的basic版本
    def __init__(self,num_units,forget_bias=1.0,input_size=None,state_is_tuple=True,activation=tanh,reuse=None):
        forget_bias:添加到forget门的偏置；reuse:在一个scope里是否重用

LSTMCell
    LSTM的高级版本
    def __init__(self,num_units,input_size=None,use_peepholes=False,cell_clip=None,initializer=None,num_proj=None,
                proj_clip=None,num_units_shards=None,num_proj_shards=None,forget_bias=1.0,state_is_tuple=True,activation=tanh,reuse=None):
        use_peepholes:默认False，True表示开启Peephole连接；cell_clip:是否输出前对cell状态按照给定值进行截断处理；initializer：指定初始化函数
        num_proj：通过projection层进行模型压缩的输出纬度；proj_clip将num_proj按照给定的proj_clip截断

GRUCell
    def __init__(self,num_units,input_size=None,activation=tanh,reuse=None)

MultiRNNCell
    def __init__(self,cells,state_is_tuple=True)：
        cells:一个cell列表，将列表中的cell一个个堆叠起来，如果使用cells=[cell1,cell2]，就是一个共有2层，数据经过cell1后还要通过cell2
        state_is_tuple:如果True则返回的是n-tuple，即将cell的输出值与cell的输出状态组成了一个tuple。其中，输出值的结构为c=[batch_size,num_units]
        输出状态的结构为h=[batch_size,num_units]

1、静态RNN构建
    def static_rnn(cell,inputs,initial_state=None,dtype=None,sequence_length=None,scope=None)
    inputs:是list或者二维张量，list的序列就是时间序列。元素就是每一个序列的值。

2、动态RNN构建
    def dynamic_rnn(cell,inputs,sequence_length=None,initial_state=None,dtype=None,parallel_iterations=None,swap_memory=False,time_major=False,scope=None):
        inputs：一个张量，一般是三维张量[batch_size,max_time,...];max_time表示时间序列总数

3、双向RNN构建
    tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length=None,initial_state_fw=None,initial_state_bw=None,dtype=None,
            parallel_iterations=None,swap_memory=False,time_major=False,scope=None)
        返回值：是一个tuple(outputs,output_state_fw,output_state_bw),outputs也是tuple(output_fw,output_bw),每一个值为一个张量[batch_size,max_time,layers_output],
                如果需要总得结果，可以将前向后项的layer_outpus使用tf.concat连接起来
    tf.contrib.rnn.static_bidirectional_rnn()
    tf.contrib.rnn.stack_birdiectional_rnn()：多层双向网络
    tf.contrib.rnn.stack_birdiectional_dynamic_rnn()：动态多层双向RNN网络