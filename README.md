# Important-Snippets

A personal collection of reusable Python / PyTorch and shell snippets for everyday deep-learning and computer-vision workflows.

## Overview

This repository is a grab-bag of small, self-contained utilities that recur across ML projects: building custom PyTorch dataloaders, normalizing / inverse-normalizing images, extracting CNN features, plotting t-SNE embeddings, parsing XML region annotations into masks, cataloguing image datasets into CSVs, and handy `bash`/`tmux` configuration. Each file is standalone and meant to be copied into a project and adapted. Many scripts contain hard-coded absolute paths (e.g. `/home/dipesh/...`) from their original use, so you will usually need to edit the paths at the top of a file before running it.

## Requirements

There is no `requirements.txt` or `environment.yml`; dependencies are inferred from the imports across the scripts:

- Python 3
- [PyTorch](https://pytorch.org/) (`torch`) and `torchvision`, with `torch.utils.tensorboard`
- `numpy`, `scipy`, `pandas`
- `opencv-python` (`cv2`)
- `Pillow` (`PIL`)
- `matplotlib`, `seaborn`
- [`MulticoreTSNE`](https://github.com/DmitryUlyanov/Multicore-TSNE) (for the t-SNE plots)
- `tqdm`
- `wget` (Python module) and/or `gdown` (for the download snippets)

Install the ones you need, for example:

```bash
pip install torch torchvision numpy scipy pandas opencv-python pillow matplotlib seaborn tqdm wget gdown MulticoreTSNE
```

## Repository layout

```
bash/            shell, tmux and download helper snippets
mnist_download/  download MNIST and export it as JPGs
plots/           t-SNE and 2D scatter plots; mask crop / convex-hull
python_basic/    XML annotation parsing, mask rasterization, dataset CSVs/lists
pytorch/         custom dataloaders, (inverse) normalization, feature extraction, models
```

## Usage

Each script is run directly with `python <script>.py`. Before running, open the file and update any hard-coded input/output paths.

### `pytorch/` — dataloaders, normalization, models

- **`imagefolder.py`** — a custom `ImageFolder`/`DatasetFolder` whose `__getitem__` returns a `(normalized, un-normalized, target)` triple, and logs the normalized, inverse-normalized, and original image to TensorBoard (log dir `img`). Expects a class-subfoldered dataset at `../test`. Run from inside `pytorch/`:
  ```bash
  cd pytorch
  python imagefolder.py
  tensorboard --logdir img
  ```
  `imagefolder_crude.py` is a script-style variant of the same idea (includes an alternative `MyCustomImageFolder`).
- **`inverse_normalization.py`** — loads a single image, applies ImageNet normalization, then inverse-normalizes it and logs both to TensorBoard. Note: the top-level line `from net import UnNormalize` refers to a `net` module that is not in this repo and will fail on import; `UnNormalize` is redefined immediately below, so remove that import line to run the script.
- **`resnet_feature_extractor.py`** — runs a pretrained `resnet18` in eval mode over `ImageFolder` datasets (a "mask"/"nomask" split) and saves the 1000-dimensional outputs as `.npy` feature files. Edit the dataset directories at the top before running.
- **`load_csv_torch.py`** — a `load_data()` helper that reads pre-extracted Office-Home CSV feature files, builds source/target `TensorDataset`/`DataLoader` objects for a domain-adaptation setup, and caps the (age) label at 25. Import and call `load_data(batch_size=...)` from your training script.
- **`net_editable_vgg.py`** — `Vgg19edited`, a VGG-19 with frozen pretrained features split into two blocks (`part1`/`part2`), an adaptive average pool, and a new `Linear` classifier head (`num_classes=14` by default). Run directly to print the model and a forward-pass output shape.
- **`image_show_dataloader.py`** — a short code fragment (not a standalone script) for plotting a grid of training images pulled from a dataloader; paste it into an existing training script or notebook that already defines `train_dataloader`, `device`, `plt`, `np`, and `vutils`.

### `plots/` — visualization

- **`tsne.py`** — loads two `.npy` feature arrays, runs MulticoreTSNE, and saves a two-class t-SNE scatter plot (`mask_non-mask_tsne.png`).
- **`tsne_four_tensors.py`** — same as above for four feature arrays / classes, saving a four-class t-SNE plot.
- **`2d_plot_four_tensors.py`** — plots four feature tensors directly as a 2D seaborn scatter (the t-SNE step is commented out; assumes already-2D inputs such as softmax outputs).
- **`crop_convexhull.py`** — crops an image to the bounding box of its binary mask and draws OpenCV contours and convex hulls. This is a rough snippet with hard-coded paths and a leftover `pdb.set_trace()`; adapt before use.

### `python_basic/` — annotations and dataset bookkeeping

- **`xml_loading.py`** — `get_annotations(path)` parses ImageScope/Aperio-style XML (`Annotation/Regions/Region/Vertices`) and returns per-annotation vertex regions, annotation IDs, and line colors.
- **`img_mask.py`** — uses `xml_loading.get_annotations` to rasterize polygon annotations into colour overlays and binary masks, saving them into id-wise output folders. Set the annotation glob (`../new_extracted/MODEL/*/*.xml`) and output paths first.
- **`make_csv.py`** — walks an image folder tree and writes `data.csv` (train/test split, class, path, width, height, size in MB) plus `num_classes.json` with per-class counts.
- **`make_list.py`** — walks a folder and writes `mask.txt` / `no_mask.txt` file lists based on the class-folder name prefix.

### `mnist_download/`

- **`download_and_save_mnist_jpgs.py`** — downloads the MNIST pickle and exports every train/val/test digit as an individual JPG under `train/`, `val/`, and `test/`. Note: the built-in download URL (`http://deeplearning.net/data/mnist/mnist.pkl.gz`) points to a site that is no longer online; supply your own `mnist.pkl.gz` or update the URL to a working mirror before running.

### `bash/` — shell configuration and recipes

Plain-text reference snippets (copy the relevant parts into your dotfiles):

- **`bashrc.txt`** — a `.bashrc` with git/tmux/venv/vpn aliases and shortcuts.
- **`tmux_conf.txt`, `tmux_conf2.txt`** — `.tmux.conf` settings (prefix remap, pane navigation, status bar).
- **`gdrive_download.txt`** — `wget`/`gdown` recipes for downloading small and large Google Drive files.
- **`git_passwd.txt`** — notes on configuring the git credential helper.

## Results / Figures

The `pytorch/results/` directory contains example TensorBoard-exported outputs from the normalization snippets: an ImageNet-normalized image tensor rendered directly (values fall outside the display range, hence the vivid colors) and the same image recovered after inverse normalization.

| Normalized tensor | After inverse normalization |
| --- | --- |
| ![Normalized image](pytorch/results/normalized_img.png) | ![Inverse-normalized image](pytorch/results/inverse_norm.png) |

## Datasets

- **MNIST** — used by `mnist_download/download_and_save_mnist_jpgs.py` (see the note above about the defunct download URL).
- **Office-Home** — `pytorch/load_csv_torch.py` expects pre-extracted per-domain CSV feature files (Art / Clipart / Product / Real-World); these feature files are not included in the repo.

## License

Released under the [MIT License](LICENSE).
