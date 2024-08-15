from setuptools import setup, find_packages

setup(
    name='ssdl_mn_v3',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch==2.3.0',  # using older versions of torch/torchvision leads to segmentation fault on Raspberry Pi 3B
        'torchvision==0.18.0',
        "opencv-python==4.10.0.82",
        'tqdm==4.66.1',
        'numpy==1.26.2',
        'pandas==2.1.3',
        'flwr==1.5.0',
        'sgmllib3k==1.0.0',
        'transforms==0.2.1',
        'matplotlib==3.8.3',
        'pycocotools==2.0.7',
        'setuptools==70.0.0',
        'wandb==0.16.4',
        'plotly==5.21.0',
        'gpiozero==2.0.1'
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
