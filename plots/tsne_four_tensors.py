import os
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
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


tsne = TSNE(n_jobs=16)

embeddings = tsne.fit_transform(all_features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
sns.set(rc={'figure.figsize':(11.7,8.27)})
NUM_CLASSES = 4
palette = sns.color_palette("bright", NUM_CLASSES)

label_dict={0:"mask_blur",
			1:"mask_noblur",
			2:"nomask_blur",
			3:"nomask_noblur"
}

hue_label = [label_dict[i] for i in dataset_label[:,0]]

# pdb.set_trace()
# plt.legend(["mask_blur", "mask_noblur", "nomask_blur", "nomask_noblur"], loc='lower right')
# marker_list = ['x','x','o','o']
plot = sns.scatterplot(vis_x, vis_y, hue=hue_label, legend='full', palette=palette)
# plot = sns.scatterplot(vis_x, vis_y, hue=hue_label, legend='full',markers=marker_list, palette=palette)
# plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], legend='brief', palette=palette)
# plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], palette=palette)

plt.savefig("mask_pred_2_mask_no-mask_blur_no-blur_keras_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))