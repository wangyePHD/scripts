# import pickle
# import numpy as np
# import random
# import os

# root = '/home/wangye/Code/3D_SD_knowledge_base/Data'
# target = '/home/wangye/Code/3D_SD_knowledge_base/Data/ModelNetFewShot'

# train_data_path = os.path.join(root, 'modelnet40_train_8192pts_fps.dat')
# test_data_path = os.path.join(root, 'modelnet40_test_8192pts_fps.dat')
# # train
# with open(train_data_path, 'rb') as f:
#     train_list_of_points, train_list_of_labels = pickle.load(f)
# with open(test_data_path, 'rb') as f:
#     test_list_of_points, test_list_of_labels = pickle.load(f)

# # list_of_points = train_list_of_points + test_list_of_points  
# # list_of_labels = train_list_of_labels + test_list_of_labels


# train_file_list = "/home/wangye/Code/3D_SD_knowledge_base/Data/ModelNet/modelnet40_normal_resampled/modelnet40_train.txt"
# test_file_list = "/home/wangye/Code/3D_SD_knowledge_base/Data/ModelNet/modelnet40_normal_resampled/modelnet40_test.txt"

# train_file = []
# test_file = []

# f = open(train_file_list,'r')
# for line in  f.readlines():
#     train_file.append(line.strip())

# f = open(test_file_list,'r')
# for line in  f.readlines():
#     test_file.append(line.strip())



# def generate_fewshot_data(way, shot, prefix_ind, eval_sample=20):
#     train_cls_dataset = {}
#     test_cls_dataset = {}
#     train_file_dataset = {}
#     test_file_dataset = {}
#     train_dataset = []
#     test_dataset = []
#     file_train = []
#     file_test = []
#     # build a dict containing different class
#     for point, label,file in zip(train_list_of_points, train_list_of_labels,train_file):
#         label = label[0]
#         if train_cls_dataset.get(label) is None:
#             train_cls_dataset[label] = []
#             train_file_dataset[label] = []
#         train_cls_dataset[label].append(point)
#         train_file_dataset[label].append(file)
#     # build a dict containing different class
#     for point, label, file in zip(test_list_of_points, test_list_of_labels, test_file):
#         label = label[0]
#         if test_cls_dataset.get(label) is None:
#             test_cls_dataset[label] = []
#             test_file_dataset[label] = []
            
#         test_cls_dataset[label].append(point)
#         test_file_dataset[label].append(file)
        
#     print(sum([train_cls_dataset[i].__len__() for i in range(40)]))
#     print(sum([test_cls_dataset[i].__len__() for i in range(40)]))
#     # import pdb; pdb.set_trace()
#     keys = list(train_cls_dataset.keys())
#     random.shuffle(keys)

#     for i, key in enumerate(keys[:way]):
#         train_data_list = train_cls_dataset[key]
#         train_file_list = train_file_dataset[key]
#         np.random.seed(116)
#         np.random.shuffle(train_data_list) 
#         np.random.seed(116)
#         np.random.shuffle(train_file_list)
#         assert len(train_data_list) > shot
#         for data in train_data_list[:shot]:
#             train_dataset.append((data, i, key))
#         for file in train_file_list[:shot]:
#             file_train.append(file)
            
#         test_data_list = test_cls_dataset[key]
#         test_file_list = test_file_dataset[key]
#         np.random.seed(116)
#         np.random.shuffle(test_data_list) 
#         np.random.seed(116)
#         np.random.shuffle(test_file_list)
#         # import pdb; pdb.set_trace()
#         assert len(test_data_list) >= eval_sample
#         for data in test_data_list[:eval_sample]:
#             test_dataset.append((data, i, key))
#         for file in test_file_list[:eval_sample]:
#             file_test.append(file)
#     print(file_train)
#     print("*"*50)
#     print(file_test)
#     random.shuffle(train_dataset)
#     random.shuffle(test_dataset)
   
    
#     dataset = {
#         'train': file_train,
#         'test' : file_test
#     }
#     save_path = os.path.join(target, f'{way}way_{shot}shot')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     with open(os.path.join(save_path, f'{prefix_ind}.pkl'), 'wb') as f:
#         pickle.dump(dataset, f)
    

# if __name__ == '__main__':
#     ways = [5, 10]
#     shots = [10, 20]
#     for way in ways:
#         for shot in shots:
#             for i in range(10):
#                 generate_fewshot_data(way = way, shot = shot, prefix_ind = i)
        #         break
        #     break
        # break



import pickle
path = "/home/wangye/Code/3D_SD_knowledge_base/Data/ModelNetFewShot/10way_20shot/0.pkl"

with open(path, 'rb') as f:
	a = pickle.load(f)

print(len(a['train']))
print(len(a['test']))
    

# path = "/home/wangye/Code/3D_SD_knowledge_base/Data/ModelNetFewshot_PointBERT/5way_10shot/0.pkl"

# print('*'*20)
# with open(path, 'rb') as f:
# 	a = pickle.load(f)

# print(a['train'])