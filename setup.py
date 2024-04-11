from setuptools import setup

setup(
    name='MooneyGen',
    version='0.1.0',
    py_modules=['train_images'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'gen_img = train_images:train_images',
        ],
    },
)