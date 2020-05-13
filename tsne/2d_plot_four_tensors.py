import os
import pdb
import numpy as np
from scipy import io
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time


start_time = time.time()


t_mask_blur_feat = "/home/dipesh/Desktop/age-gender/npy_feats/mask_pred_2/x_mask_blur_2_softmax.npy"
t_mask_noblur_feat = "/home/dipesh/Desktop/age-gender/npy_feats/mask_pred_2/x_mask_noblur_2_softmax.npy"
t_nomask_blur_feat = "/home/dipesh/Desktop/age-gender/npy_feats/mask_pred_2/x_nomask_blur_2_softmax.npy"
t_nomask_noblur_feat = "/home/dipesh/Desktop/age-gender/npy_feats/mask_pred_2/x_nomask_noblur_2_softmax.npy"

mask_blur_feat = np.load(t_mask_blur_feat)
mask_noblur_feat = np.load(t_mask_noblur_feat)
nomask_blur_feat = np.load(t_nomask_blur_feat)
nomask_noblur_feat = np.load(t_nomask_noblur_feat)

print("mask_blur_feat shape {}".format(mask_blur_feat.shape)) #(1001, 96, 96, 3)
print("mask_noblur_feat shape {}".format(mask_noblur_feat.shape)) #(3310, 96, 96, 3)
print("nomask_blur_feat shape {}".format(nomask_blur_feat.shape)) #(957, 96, 96, 3)
print("nomask_noblur_feat shape {}".format(nomask_noblur_feat.shape)) #(3346, 96, 96, 3)


all_features = np.concatenate((mask_blur_feat, mask_noblur_feat, nomask_blur_feat, nomask_noblur_feat), axis = 0)

dataset_label = np.zeros((all_features.shape[0],1))
dataset_label[mask_blur_feat.shape[0]:mask_blur_feat.shape[0]+ mask_noblur_feat.shape[0]] = 1
dataset_label[mask_blur_feat.shape[0]+ mask_noblur_feat.shape[0]:mask_blur_feat.shape[0]+ mask_noblur_feat.shape[0]+nomask_blur_feat.shape[0]] = 2
dataset_label[mask_blur_feat.shape[0]+ mask_noblur_feat.shape[0]+nomask_blur_feat.shape[0]:] = 3


# tsne = TSNE(n_jobs=16)

# embeddings = tsne.fit_transform(all_features)

# vis_x = embeddings[:, 0]
# vis_y = embeddings[:, 1]
sns.set(rc={'figure.figsize':(11.7,8.27)})
NUM_CLASSES = 4
palette = sns.color_palette("bright", NUM_CLASSES)

label_dict={0:"mask_blur",
			1:"mask_noblur",
			2:"nomask_blur",
			3:"nomask_noblur"
}

hue_label = [label_dict[i] for i in dataset_label[:,0]]

# plot = sns.scatterplot(data = (mask_blur_feat[:,0], mask_blur_feat[:,1]), x="no_mask",y="mask", legend='full', color="r",marker='X',s=100)
plot = sns.scatterplot(mask_blur_feat[:,0], mask_blur_feat[:,1]+0.01, legend='full', color="r",marker='X',s=100)
plot = sns.scatterplot(mask_noblur_feat[:,0], mask_noblur_feat[:,1]+0.02, legend='full', color="b",marker='X',s=200)
plot = sns.scatterplot(nomask_blur_feat[:,0], nomask_blur_feat[:,1]-0.01, legend='full', color="g",marker='o',s=100)
plot = sns.scatterplot(nomask_noblur_feat[:,0], nomask_noblur_feat[:,1]-0.02, legend='full', color="y",marker='o',s=100)

plt.savefig("mask_pred_2_mask_no-mask_blur_no-blur_keras_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))