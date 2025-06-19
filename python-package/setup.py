from pathlib import Path
from setuptools import setup, find_packages

cwd = Path(__file__).resolve().parent
requirements = (cwd / 'fashion_pipeline' /
                'requirements.txt').read_text().split('\n')


setup(
    name='fashion_pipeline',
    version='0.1.0',
    description="""A machine learning pipeline
        for product review classification.""",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'': ['./data/reviews.csv',
                       './data/gb_model.pkl',
                       './data/lr_model.pkl',
                       'requirements.txt']},
    install_requirements=requirements,
)
