cd notebooks

bash ./check_notebooks.sh wasserstein_classification_MNIST08
bash ./check_notebooks.sh wasserstein_toy_classification
bash ./check_notebooks.sh wasserstein_toy
bash ./check_notebooks.sh wasserstein_classification_fashionMNIST

cd -

make html
