.PHONY: help prepare-dev test doc ipynb-to-rst
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py36, py37, py38"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make doc"
	@echo "       build Sphinx docs documentation"
	@echo "ipynb-to-rst"
	@echo "       Transform notebooks to .rst files in documentation and generate the doc"
prepare-dev:
	python3 -m pip install virtualenv
	python3 -m venv torchlip_dev_env
	. torchlip_dev_env/bin/activate && pip install --upgrade pip
	. torchlip_dev_env/bin/activate && pip install -r requirements.txt
	. torchlip_dev_env/bin/activate && pip install -r requirements_dev.txt	

test:
	. torchlip_dev_env/bin/activate && tox 

test-disable-gpu:
	. torchlip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

doc:
	. torchlip_dev_env/bin/activate && cd docs && make html && cd -

ipynb-to-rst:
	. torchlip_dev_env/bin/activate && cd docs && ./generate_doc.sh && cd -
