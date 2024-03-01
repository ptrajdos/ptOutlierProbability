from setuptools import setup, find_packages



setup(
        name='pt_outlier_probability',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/ptOutlierProbability',
        description="Objects for expressing outlier scores as probabilities",
        packages=find_packages(include=[
                'pt_outlier_probaiblity',
                'pt_outlier_probaiblity.*',
                ]),
        install_requires=[ 
                'numpy>=1.22.4',
                'scikit-learn>=1.2.2',
        ],
        test_suite='test'
        )
