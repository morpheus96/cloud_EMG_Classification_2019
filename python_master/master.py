import os, io
os.environ['KERAS_BACKEND']='theano'
import sys
import numpy as np
import keras
from keras.layers import Reshape, Conv1D, BatchNormalization, LeakyReLU, Flatten, Dropout, Dense
import paramiko as ssh

def count_nn_cloud(ip, password, name):
    knn= keras.Sequential()
    knn.add(Reshape((620, 1), input_shape=(620,)))
    knn.add(Conv1D(80, 5 , padding='same'))
    knn.add(LeakyReLU(0.2))
    knn.add(Conv1D(60, 5 , padding='same'))
    knn.add(LeakyReLU(0.2))
    knn.add(Conv1D(55, 5 , padding='same'))
    knn.add(LeakyReLU(0.2))
    knn.add(Conv1D(50, 5 , padding='same', strides=2))
    knn.add(LeakyReLU(0.2))
    knn.add(BatchNormalization())

    knn.add(Flatten())
    knn.add(Dense(40))
    knn.add(LeakyReLU(0.3))
    knn.add(Dropout(0.05))
    knn.add(Dense(5, activation='sigmoid'))

    knn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

    print("KNN Defined")
    host_name=ip
    host_password=password
    print(host_name)
    print(host_password)
    print("Attempting to establish ssh connection")
    slave=ssh.SSHClient()
    slave.set_missing_host_key_policy(ssh.AutoAddPolicy)
    slave.connect(host_name, port=2222, username="root", password=host_password)
    print("SSH connection established \n Attemping to establish SFTP channel")
    slave_sftp=slave.open_sftp()
    knn.save('architecture_{0}.h5'.format(name))
    slave_sftp.put('architecture_{0}.h5'.format(name), '/architecture.h5')
    knn_trained=None
    try:
        slave.exec_command("python /slave.py")
        slave_sftp.get('/trained.h5', 'trained_{0}.h5'.format(name))
        #knn_trained=keras.models.load_model('trained.h5')
        slave_sftp.remove('/trained.h5')
    except:
        print("Achtung - transfer not performed")
    slave.close()

print( '\n{0}\n{1}\n{2}\n{3}\n'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) )
count_nn_cloud(sys.argv[1], sys.argv[2], 'first')
count_nn_cloud(sys.argv[3], sys.argv[4], 'first')
print("networks counted")