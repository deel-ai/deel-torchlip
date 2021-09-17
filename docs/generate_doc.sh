cd notebooks
bash ./check_notebooks.sh wasserstein_classification_MNIST08
bash ./check_notebooks.sh wasserstein_toy_classification
bash ./check_notebooks.sh wasserstein_toy

cd -

sphinx-apidoc --implicit-namespaces -f -e -o source ../deel

make html
