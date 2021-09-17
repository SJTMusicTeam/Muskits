# If a Python interpreter is specified, then creates a virtualenv from it
# Refactored from ESPNet (author: Jiatong Shi)

# Disable cupy installation
NO_CUPY :=
# PyTorch version: 1.7.0, 1.7.1, 1.8.1
TH_VERSION := 1.8.1
WGET := wget --tries=3

# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
IS_CONDA := 0
USE_PIP :=
else
IS_CONDA :=
USE_PIP := 0
endif


# Set if install binaries on CPU mode e.g. make CPU_ONLY=1
# If you don't have any GPUs, this value will be set automatically
ifeq ($(shell which nvcc),) # 'nvcc' not found
CPU_ONLY := 0
else
CPU_ONLY :=
endif

ifeq ($(strip $(CPU_ONLY)),)
# Derive CUDA version from nvcc
CUDA_VERSION = $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(CUDA_VERSION)))
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch_stable.html
ifeq ($(strip $(CUDA_VERSION_WITHOUT_DOT)),80)
CONDA_PYTORCH := pytorch=$(TH_VERSION) cuda$(CUDA_VERSION_WITHOUT_DOT)
else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION)
endif

else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cpuonly
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cpu/torch_stable.html
NO_CUPY := 0
endif


ifeq ($(shell expr $(CUDA_VERSION_WITHOUT_DOT) \>= 102), 1)
# cupy==6.0.0 doesn't support CUDA=10.2 or later
NO_CUPY := 0
endif


.PHONY: all clean

all: kaldi showenv python conda_packages.done check_install

python: activate_python.sh muskit.done pytorch.done pyopenjtalk.done


kaldi:
	test -f kaldi/egs/wsj/s5/utils/parse_options.sh || git clone --depth 1 https://github.com/kaldi-asr/kaldi

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
	. ./activate_python.sh && python3 check_pytorch_cuda_compatibility.py $(TH_VERSION) ${CUDA_VERSION} || { echo "Change TH_VERSION or CUDA_VERSION"; exit 1; }
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo IS_CONDA=$(IS_CONDA)
	@echo USE_PIP=$(USE_PIP)
	@echo TH_VERSION=$(TH_VERSION)
	@echo CONDA_PYTORCH=$(CONDA_PYTORCH)
	@echo PIP_PYTORCH=$(PIP_PYTORCH)
	@echo NO_CUPY=$(NO_CUPY)

#########################################

cmake.done: activate_python.sh
	. ./activate_python.sh && { command -v cmake || conda install -y cmake; }
	touch cmake.done
flac.done: activate_python.sh
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	touch flac.done
ffmpeg.done: activate_python.sh
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
	touch ffmpeg.done
sox.done: activate_python.sh
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	touch sox.done
sndfile.done: activate_python.sh
	. ./activate_python.sh && { python3 -c "from ctypes.util import find_library as F; assert F('sndfile') is not None" || conda install -y libsndfile=1.0.28 -c conda-forge; }
	touch sndfile.done
ifneq ($(strip $(IS_CONDA)),)
conda_packages.done: cmake.done flac.done ffmpeg.done sox.done sndfile.done
else
conda_packages.done:
endif
	touch conda_packages.done


ifneq ($(strip $(USE_PIP)),)
pytorch.done: activate_python.sh
	. ./activate_python.sh && python3 -m pip install numpy --upgrade  # Workaround for "ImportError: numpy.core.multiarray failed to import"
	. ./activate_python.sh && python3 -m pip install $(PIP_PYTORCH)
	touch pytorch.done
else
pytorch.done: activate_python.sh
	. ./activate_python.sh && conda install -y $(CONDA_PYTORCH) -c pytorch
	touch pytorch.done
endif

# NOTE(kamo): conda_packages is not necessary for installation of muskit, but add it the dependencies just in case.
muskit.done: pytorch.done conda_packages.done
	. ./activate_python.sh && python3 -m pip install -e "..[recipe]"  # Install editable mode by default
	touch muskit.done

pyopenjtalk.done: muskit.done conda_packages.done
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	touch pyopenjtalk.done

check_install: python
ifeq ($(strip $(CPU_ONLY)),)

ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh; python3 check_install.py --no-cupy
else
	. ./activate_python.sh; python3 check_install.py
endif

else

ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh; python3 check_install.py --no-cuda --no-cupy
else
	. ./activate_python.sh; python3 check_install.py --no-cuda
endif

endif


clean: clean_extra
	rm -rf pyopenjtalk.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -f muskit.done pytorch.done
	find . -iname "*.pyc" -delete

