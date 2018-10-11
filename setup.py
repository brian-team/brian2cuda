from setuptools import setup, find_packages

setup(
    name='brian2cuda',
    version='dev',
    packages=find_packages(),
    package_data={# include template files
                  'brian2cuda': ['templates/*.cu',
                                 'templates/*.h',
                                 'templates/makefile',
                                 'templates/win_makefile',
                                 'brianlib/*.cu',
                                 'brianlib/*.h'],
                 },
)
