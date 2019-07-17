import tensorflow as tf

#Func:tf.ones tf.zeros
sess =tf.InteractiveSession()
x=tf.ones([3,4,5],tf.float32)#三维张量
print("3 dimension  tensor:")
print(sess.run(x))
x=tf.ones([3,4],tf.float32)  #二维张量
print("2 dimension tensor:")
print(sess.run(x))