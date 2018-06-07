from distutils.core import setup

setup(name='Duke',
    version='1.2.0',
    description='Tabular Dataset Summarization System',
    packages=['Duke'],
    install_requires=['pandas >= 0.19.2',
        'numpy>=1.13.3',
        'gensim==3.2.0',
        'inflection>=0.3.1',
        'ontospy>=1.8.6'],
    include_package_data=True,
)