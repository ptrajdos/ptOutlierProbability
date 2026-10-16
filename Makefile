ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))

SRCDIR=${ROOTDIR}/pt_outlier_probability
TESTDIR=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov
COVERAGERC=${ROOTDIR}/.coveragerc

INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv

DOCS_DIR=${ROOTDIR}/docs
TOXDIR=${ROOTDIR}/.tox

COVERAGE = coverage
UNITTEST_PARALLEL = unittest-parallel
PDOC= pdoc3
PYTHON=python
SYSPYTHON=python
PIP=pip
PYTEST=pytest
VENV_OPTIONS=
TOX=tox
TOX_CORES=auto


LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log

ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif

.PHONY: all clean clean_pypackages clean_venv clean_tox test test_parallel docs profile tox_check

all: profile

clean: clean_pypackages clean_venv clean_tox
	@echo "Cleaning up build artifacts, virtual environments, and test logs..."

clean_pypackages:
	rm -rf pypackages

clean_venv:
	rm -rf ${VENV_SUBDIR}

clean_tox:
	rm -rf ${TOXDIR}

venv:
	${SYSPYTHON} -m venv --upgrade-deps ${VENV_OPTIONS} ${VENV_SUBDIR}

pypackages: venv setup.py
	${ACTIVATE}; ${PYTHON} -m ${PIP} install -e "${ROOTDIR}[test]" --prefer-binary --log ${INSTALL_LOG_FILE}
	touch $@

test: pypackages
	mkdir -p ${LOGDIR}
	${ACTIVATE}; ${COVERAGE} run --rcfile=${COVERAGERC} --branch --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} -t ${ROOTDIR} 2>&1 |tee -a ${LOGFILE}
	${ACTIVATE}; ${COVERAGE} html --rcfile=${COVERAGERC} --show-contexts -d ${COVDIR}

test_parallel: pypackages
	mkdir -p ${COVDIR} ${LOGDIR}
	${ACTIVATE}; ${UNITTEST_PARALLEL} --level class -v -t ${ROOTDIR} -s ${TESTDIR} -p '*_test.py' --coverage --coverage-rcfile ${COVERAGERC} --coverage-source ${SRCDIR} --coverage-html ${COVDIR} 2>&1 |tee -a ${LOGFILE}

docs: pypackages
	${ACTIVATE}; $(PDOC) --force --html ${SRCDIR} --output-dir ${DOCS_DIR}

profile: pypackages
	${ACTIVATE}; ${PYTEST} -n auto --cov-report=html:${COVDIR} --cov=${SRCDIR} --profile ${TESTDIR}

tox_check: pypackages
	${ACTIVATE}; ${TOX} -p ${TOX_CORES}
