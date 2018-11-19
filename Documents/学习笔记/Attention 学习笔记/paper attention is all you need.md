# 	Attention is all your need  

5:05  **Introduction**

**Attention is a mechnism here to basically imporve network performance** .Attention will do in this particular case if we look at the decoder here.If it is trying to predict this word for cat or the next word here. In essence,the only information we have is what the last word was generated for cat and what the hidden state is .



**Campared with  regular RNN:**

So if we look at what network should output in the input sentence(eats) .

And if we look at kind of the information flow,that is  the word has travelled.  Firstly, it means to encode toword word vector. Same function for all the word. 

It goes through this hidden state and again into another state. because we have two more tokens and next hidden state.  and then they goes all the way to the decorder.where the first two words are decoded.  H6 is the hidden state somehow still need to retain the information that now the word eats is word to be translated,  decoder should find german word for that.That will be a long path , a lot of translation involved.Hidden state need to memorize all the structure and scentence , all the word for rnn.

It's very hard for rnn to learn long range denpendency.  Naturally you just think oh I can decode the first word to the first word, then second word to the second word. Actually the word pretty well in this example. Die for the , katze for cat decode one by one.



**Attention is a mechnism by the decoder here in the step which we look here .That is decided to go back and look at a particular part of input.**

8:22

The decoder can decide to attend to the hidden of  input scentences.

Example, Teach decoder to **look particular hidden state（h2）** , to pay close attention to certain step here. because that was step when **the word "eats"** is just encoded ,so it probably has a lot of information about **what i would like to do now to translate word eat**.

The word Path length of information is much shorter.  Word vector go through one enconding step ,then one hidden step, then one decoding step. The decoder can look directly for that.



**Decoder function:**

Decoder would output bunch of keys  [k1... kn] . keys will index kind of hidden state 

kind of softmax architecture and we gonna not look at this ,the actual paper gonna be clear.

Decoder is designed to attend the input scentence to draw inforamtion directly from there.



**Attention is all you need , you don't need entire recurrent thing.** 

You just do attention over every thing, it will be fine.Two part, we called  encoder and decoder. 



11分钟左右

Propose the transformer architecture:

This all happened at once.     

**We would feed the entire source sentence and  the target sentence** we produce so far to this network.  source sentence will go into this part and  target we produce so far will go into this part . **And this is all combined**, at the end we get the output here.The output probabilities that kind tells us the probablity to for the that next word.  so we can choose the top probability and then repeat the entire process. and ever step in procution is one training sample.  

Here before the rnn, the entire training sample is one sample, because we need to backpropagate for all these rnn steps cause they all happen in sequence.

Here, basically, output of one single token is one sample and then computation finish the what will happen for everything, only for this one step. there's no more teeth step for kind of backpropagation as  rnn.  

one step prediction 



13:40

paradigm shift 



input embedding and ouput embedding are semantical,basically the token can be embeded with vector here. 

**Positional encoding** is kind of special thing. To prevent from losing sequence information of output.  Encode the kind of wired word you push through the network.The network can recongize the sequence of each word.Compared the words.

Encode all the sequencies with the maximum frequency sequence, encode every change step of them. 

Compare two words, look at all the skills of the things.  Most of all ,the curve.



**Attention:** 

Attention part : 

with  multi-Head Attention  and Add&Norm part.

Three attention parts in graph.

**Bottom left attention** is aim at input sentence. 1. encoded into hidden representation, it happened all at once.you put together this hidden representation and all you do is to use the attention of the input sequence. Like picking up which word you use more or less.



So with the **bottom right** , the output scentence is encoded into hidden state, the process is like input attention.



The **top right attention** is most interesting part ot attention,  unites the encoder part with the decoder part. **It combines the source sentence with the target sentence.** 

As  you can see here , there's output going from the part of encoded source sentence and it goes into the multi-head attention, there are two connections . There is also one connection coming from the encoded output so far. Total three connections going to this multi-head connection.



18:44 时间

Three connections are **（K）keys, (V) values and (Q）Queries.**

V, K are output by the encoding part of source scentence and Q are output by the encoding part of target scentence. 

These are not one key,value  or query. There are many kind of multi-head attention instead of one. There are a set of queries,keys,values simultaneously, attention compute here :

​                 $Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})$V



if you **dot product the keys and queries**, the dot product give you the angel of two vectors,

especially high dimension, most vectors are going to kind of 90 degrees, most vectors are not aligned very well , so their dot product will be zero, but if the key and query will align each other,

pointed to the same direction the product will be large here.

Each key has an  associate value,  there are table  index is key , value is value.

When we introduce a query, what we will do, query will be a vector like this. 

Query dot product with each of keys, then we compute the softmax, so one key will be selected with highest probability. 



Multiply the result with values, $softmax(<k_{2},Q>)$  , multiply the values , it will basically select one value too.  This is the network will compute into further things, so you see the output there going to more networks.



**Conlusion:**   23分钟

Whole process, encoder of input source discover the interesting things about the source scentens and build key value pairs.

The encoder of output scentence build queries.  

**For values,.here is a bunch of things for source scentence you may find interesting ,keys are ways to index the values. Queries are that I would like to know certain things.A scope to find the thing I need know** 

For example , value will be the name of person,  the key is index like name  height weight. query is like what do I want, I want the name . The key will be name and corresponding value will be the certain person.



 The point made for this attention is reducing the path lengths. Reducing the amount of computation stacks. 



https://github.com/tensorflow/tensor2tensor    Train and evaluate model for attention.





