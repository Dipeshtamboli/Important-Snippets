"""
This code shows some images from trainng samples directly loading from train loader
"""

#Plot some training images
train_batch = next(iter(train_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(train_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))