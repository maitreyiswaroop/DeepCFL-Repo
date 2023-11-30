import torch
import numpy as np
import hyperparams

def get_MNIST_one_one_data():
    # Loading the datasets
    x_data = torch.load('./dataset/x_data.pt')
    y_data = torch.load('./dataset/y_data.pt')
    class_data = torch.load('./dataset/class_data.pt')
    print(x_data.shape,y_data.shape,class_data.shape)

    x_data = x_data.to(torch.float32)
    y_data = y_data.to(torch.float32)
    class_data = class_data.to(torch.float32)
    x_labels = x_data[:, -1]
    x_labels = x_labels.to(torch.long)
    x_data = x_data[:, :-1]

    if len(x_data.shape)<3:
        if x_data.shape[1]>28*28:
            x_data=x_data[:,:28*28]
        x_data=x_data.view(x_data.shape[0],1,28,28)

    # Shuffling the datasets
    n = x_data.shape[0]

    # Generating random permutation indices
    indices = torch.randperm(n)

    # Shuffling the datasets using the same indices, splitting into train and test
    x_train = x_data[indices[:int(0.8*n)]]
    y_train = y_data[indices[:int(0.8*n)]]
    class_train = class_data[indices[:int(0.8*n)]]
    train_labels = x_labels[indices[:int(0.8*n)]]

    x_test = x_data[indices[int(0.8*n):]]
    y_test = y_data[indices[int(0.8*n):]]
    class_test = class_data[indices[int(0.8*n):]]
    test_labels = x_labels[indices[int(0.8*n):]]

    # obtaining a sample of each digit
    idx_sample=[0]*10
    for j in range(10):
        idx_sample[j]=np.random.choice([i for i in range(0,x_test.shape[0]) if test_labels[i]==j])

    sample_digits=[[]]*10
    for j in range(10):
        sample_digits[j]=np.random.choice([i for i in range(0,x_test.shape[0]) if test_labels[i]==j],10)

    sample_digits=[item for sublist in sample_digits for item in sublist]
    return x_train, y_train, class_train, train_labels, x_test, y_test, class_test, test_labels, idx_sample, sample_digits

def get_MNIST_many_one_data():
    """
        Generates a dataset where all the X digits are all the MNIST digits 
        and the corresponding Y digits are the Kannada digits with label 1 for all the odd X digits and label 2 for all the even X digits
    """
    # Loading the datasets
    x_data = torch.load('./dataset/x_data.pt')
    y_data = torch.load('./dataset/y_data.pt')
    class_data = torch.load('./dataset/class_data.pt')
    print(x_data.shape,y_data.shape,class_data.shape)

    x_data = x_data.to(torch.float32)
    y_data = y_data.to(torch.float32)
    class_data = class_data.to(torch.float32)
    x_labels = x_data[:, -1]
    x_labels = x_labels.to(torch.long)
    x_data = x_data[:, :-1]

    if len(x_data.shape)<3:
        if x_data.shape[1]>28*28:
            x_data=x_data[:,:28*28]
        x_data=x_data.view(x_data.shape[0],1,28,28)

    # tensor of shape y_data
    y_data_new = y_data.clone()

    # Obtaining all the indices with label 1 in the Kannada dataset
    indices = torch.where(x_labels==1)[0]
    # print(indices.shape)
    # list of indices of all odd digits
    target_indices = []
    for k in [3,5,7,9]:
        target_indices.append(torch.where(x_labels==k)[0])
    
    for idx in target_indices:
        r_idx = np.random.choice(indices, size=1)
        y_data[idx] = y_data_new[r_idx].clone()

    # Obtaining all the indices with label 2 in the Kannada dataset
    indices = torch.where(x_labels==2)[0]
    # print(indices.shape)
    # list of indices of all odd digits
    target_indices = []
    for k in [0,4,6,8]:
        target_indices.append(torch.where(x_labels==k)[0])
    
    for idx in target_indices:
        r_idx = np.random.choice(indices, size=1)
        y_data[idx] = y_data_new[r_idx].clone()

    # deleting y_data
    del y_data_new
    # Shuffling the datasets
    n = x_data.shape[0]
    # Generating random permutation indices
    indices = torch.randperm(n)

    # Shuffling the datasets using the same indices, split into train and test
    x_train = x_data[indices[:int(0.8*n)]]
    y_train = y_data[indices[:int(0.8*n)]]
    class_train = class_data[indices[:int(0.8*n)]]
    train_labels = x_labels[indices[:int(0.8*n)]]

    x_test = x_data[indices[int(0.8*n):]]
    y_test = y_data[indices[int(0.8*n):]]
    class_test = class_data[indices[int(0.8*n):]]
    test_labels = x_labels[indices[int(0.8*n):]]


    # obtaining a sample of each digit
    idx_sample=[0]*10
    for j in range(10):
        idx_sample[j]=np.random.choice([i for i in range(0,x_test.shape[0]) if test_labels[i]==j])

    sample_digits=[[]]*10
    for j in range(10):
        sample_digits[j]=np.random.choice([i for i in range(0,x_test.shape[0]) if test_labels[i]==j],10)

    sample_digits=[item for sublist in sample_digits for item in sublist]
    return x_train, y_train, class_train, train_labels, x_test, y_test, class_test, test_labels, idx_sample, sample_digits