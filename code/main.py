from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)

    # ------------Kernel Logistic Regression Case------------
    # Run your kernel logistic regression model here
    learning_rate = 0.001
    max_epoch = 40
    batch_size = 64
    # sigmas = [1,5,10,15,20,25,30]
    sigma = 20
    # scores = []
    # for sigma in sigmas:
    #     model = Model('Kernel_LR', train_valid_X.shape[0], sigma)
    #     model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    #     scores.append(model.score(test_X, test_y))
    # for i in range(len(sigmas)):
    #     print("Kernal Logistic Regression Model{:d} and {:.4f}\\\\".format(sigmas[i], scores[i]*100))

    # model = Model('Kernel_LR', train_X.shape[0], sigma)
    # model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model = Model('Kernel_LR', train_valid_X.shape[0], sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))

    # ------------RBF Network Case------------
    # Run your radial basis function network model here
   
    learning_rate = 0.001
    max_epoch = 40
    batch_size = 64
    
    sigma = 20
    
    # hidden_dim = [1,5,10,15,20,25,30,40]
    hidden_dim = 15
    # scores = []
    # for hd in hidden_dim:
    #     model = Model('RBF', hd, sigma)
    #     model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)
    #     scores.append(model.score(test_X, test_y))
    # for i in range(len(hidden_dim)):
    #     print("RBF Network score at hidden dim {:d} is {:.4f}\\\\".format(hidden_dim[i], scores[i]*100))

    # model1 = Model('RBF', hidden_dim, sigma)
    # model1.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model1 = Model('RBF', hidden_dim, sigma)
    model1.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score1 = model1.score(test_X, test_y)
    print("RBF score = {} in test set.\n".format(score1))


    # ------------Feed-Forward Network Case------------

    # Run your feed-forward network model here
  
    learning_rate = 0.001
    max_epoch = 40
    batch_size = 64
    
    # hidden_dim = [1,5,10,15,20,25,30,40]
    hidden_dim = 15
    # scores = []
    # for hd in hidden_dim:
    #     model = Model('FFN', hd)
    #     model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)
    #     scores.append(model.score(test_X, test_y))
    # for i in range(len(hidden_dim)):
    #     print("Feed forward Network at Hidden dimension {:d} is {:.4f}\\\\".format(hidden_dim[i], scores[i]*100))

    # model2 = Model('FFN', hidden_dim)
    # model2.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model2 = Model('FFN', hidden_dim)
    model2.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score2 = model2.score(test_X, test_y)
    print("Feed forward score = {} in test set\n".format(score2))
    
if __name__ == '__main__':
    main()