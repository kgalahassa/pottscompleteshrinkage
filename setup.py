from setuptools import setup, find_packages
  
with open('requirements.txt') as f:
    requirements = f.readlines()
  
long_description = 'Package for Potts Clustering with Complete Shrinkage'
  
setup(
        name ='pottscompleteshrinkage',
        version ='1.0.0',
        author ='Alahassa, Nonvikan Karl-Augustt and Alejandro, Murua',
        author_email ='alahassa@dms.umontreal.ca',
        url ='https://github.com/Vibhu-Agarwal/vibhu4gfg',
        description ='Potts Clustering with Complete Shrinkage',
        long_description = long_description,
        long_description_content_type ="text/markdown",
        license ='GNU Public Licence v3',
        packages = find_packages(),
        entry_points ={
            'console_scripts': [
                'pottscompleteshrinkage = pottscompleteshrinkage.pottscompleteshrinkage:main'
            ]
        },
        classifiers =(
            "Programming Language :: Python :: 3",
            "License :: GNU Public Licence v3",
            "Operating System :: OS Independent",
        ),
        keywords ='Potts models, Clustering, Complete Shrinkage',
        install_requires = requirements,
        zip_safe = False
)