.PHONY: help prepare-dev test doc ipynb-to-rst
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py36, py37, py38"
	@echo "make check_all"
	@echo "       check all files using pre-commit tool"
	@echo "make updatetools"
	@echo "       updatetools pre-commit tool"
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
	. torchlip_dev_env/bin/activate && pip install -r requirements.txt -f  https://download.pytorch.org/whl/cu113/torch_stable.html
	. torchlip_dev_env/bin/activate && pip install -r requirements_dev.txt
	. torchlip_dev_env/bin/activate && pip install -e .
	. torchlip_dev_env/bin/activate && pre-commit install
	. torchlip_dev_env/bin/activate && pre-commit install-hooks
	. torchlip_dev_env/bin/activate && pre-commit install --hook-type commit-msg


test:
	. torchlip_dev_env/bin/activate && tox

check_all:
	. torchlip_dev_env/bin/activate && pre-commit run --all-files

updatetools:
	. torchlip_dev_env/bin/activate && pre-commit autoupdate

test-disable-gpu:
	. torchlip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

doc:
	. torchlip_dev_env/bin/activate && cd docs && make html && cd -

ipynb-to-rst:
	. torchlip_dev_env/bin/activate && cd docs && ./generate_doc.sh && cd -
