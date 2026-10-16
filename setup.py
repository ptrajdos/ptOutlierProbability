import os
import re
from setuptools import setup, find_packages


def get_version():
        init_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'pt_outlier_probability', '__init__.py')
        with open(init_path, encoding='utf-8') as f:
                match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.M)
        if match is None:
                raise RuntimeError('Unable to find __version__ in {}'.format(init_path))
        return match.group(1)


setup(
        name='pt_outlier_probability',
        version=get_version(),
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/ptOutlierProbability',
        description="Objects for expressing outlier scores as probabilities",
        packages=find_packages(include=[
                'pt_outlier_probability',
                'pt_outlier_probability.*',
                ]),
        install_requires=[ 
                'scikit-learn>=1.2.2',
        ],
        extras_require={
                'test': [
                        'coverage',
                        'unittest-parallel>=1.6.0',
                        'pytest',
                        'pytest-xdist',
                        'pytest-cov',
                        'pytest-profiling',
                        'pdoc3',
                        'tox',
                ],
        },
        test_suite='tests'
        )
