from setuptools import setup, find_packages

setup(
    name='ssdl_mn_v3',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch == 2.3.0',
        'torchvision == 0.18.0',
        'opencv-python == 4.8.1.78',
        'tqdm == 4.66.1',
        'numpy == 1.26.2',
        'pandas == 2.1.3',
        'flwr == 1.5.0',
        'sgmllib3k',
        'transforms',
        'matplotlib',
        'pycocotools',
        'setuptools',
        'wandb'

    ],

    author='Petar Krstevski',
    author_email='krstevski1petar@gmail.com',
    description='SSDLite320-MobileNetV3-object-detection',
    long_description=open('README.md').read(),
    url='https://github.com/krstevskipetar/SSDLite320-MobileNetV3-object-detection',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
