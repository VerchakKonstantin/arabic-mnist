path_train_x = 'dataset/csvTrainImages 13440x1024.csv'
path_train_y = 'dataset/csvTrainLabel 13440x1.csv'
path_test_x = 'dataset/csvTestImages 3360x1024.csv'
path_test_y = 'dataset/csvTestLabel 3360x1.csv'
path_log = 'model_training.log'
path_model = 'model.h5'
path_predict_plot = 'predict_plot.png'
path_to_image = 'dataset/detect-1.png'

arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                     'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                     'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

input_shape = (32, 32, 1)
batch_size = 24
epochs = 50
