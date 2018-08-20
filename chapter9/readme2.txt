Encoder-Decoder框架：
    工作机制：
        先使用Encoder将输入编码映射到语义空间(通过Encoder网络生成的特征向量)，得到一个固定维数的向量，这个向量就表示输入的语义；
        然后再使用Decoder将这个语义向量解码，获得需要的输出。
    有两个输入：
        一个是x输入作为Encoder的输入，另一个是y输入作为Decoder输入，x和y依次按照各自的顺序传入网络
        标签y即参与计算loss,又参与节点运算。
        在Encoder和Decoder之间的C节点就是码器Encoder输出的解码向量，将它作为解码Decoder中cell的初始状态，进行对输出解码。
    机制优点：
        1、非常灵活，并不限制Encoder、Decoder使用何种神经网络，也不限制输入和输出的内容
        2、这是一个端到端的过程，将语义理解和语言生成合成在一起，而不是分开处理
    旧接口Seq2Seq函数：
        tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs,decoder_inputs,cell,dtype=dtype.float32,scope=None)
        encoder_inputs:一个形状为[batch_size x input_size]的list
        返回值：outputs和state。outputs为[batch_size,output_size]的张量；state为[batch_size,cell.state_size];
        cell.state_size可以表示一个或者多个子cell的状态，视输入参数cell而定。