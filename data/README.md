# Oxford-IIIT Pet dataset

Oxford-IIIT Pet is a 37-breed image dataset by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar. This project uses the species labels (cat = 0, dog = 1). Oxford distributes the dataset under **CC BY-SA 4.0** for commercial and research use and states that image copyright remains with the original owners. [Oxford VGG dataset and licence](https://www.robots.ox.ac.uk/~vgg/data/pets/)

| Split | Images | Cats | Dogs |
| --- | ---: | ---: | ---: |
| Official trainval | 3,680 | 1,188 | 2,492 |
| Project training at seed 42 | 3,128 | 1,016 | 2,112 |
| Project validation at seed 42 | 552 | 172 | 380 |
| Official test | 3,669 | 1,183 | 2,486 |

Official species counts follow the breed counts in the paper. The training and validation counts use the project’s seeded partition. The classifier stage exports the selected test cohort and effective split counts to `dataset_splits.csv`. The VAEs use the same training/validation boundary. [Parkhi et al., Cats and Dogs, Table 1](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/parkhi12cat.pdf)

## Download

Running `python -m src.train` downloads and extracts the archives with Torchvision. For a manual install, get the images and annotations from Oxford. Unpack them so your configured data root contains `oxford-iiit-pet/images/` and `oxford-iiit-pet/annotations/`, including `trainval.txt` and `test.txt`. The dataset is roughly 800 MB before extraction. [Official archives and Academic Torrents mirror](https://www.robots.ox.ac.uk/~vgg/data/pets/#downloads)

For local verification, the dataset was recovered from the fast.ai mirror. Both split files matched Oxford’s official annotations and every annotated image path was verified. Git excludes the dataset files. [fast.ai dataset archive](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz)
