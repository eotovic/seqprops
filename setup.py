from setuptools import setup, find_packages


setup(
    name='seqprops',
    version='1.0.1',
    license='GPL3',
    author="Erik Otović",
    author_email='erik.otovic@gmail.com',
    url='https://github.com/eotovic/seqprops',
    keywords='sequential properties physicochemical machine learning peptides proteins',
    install_requires=[
          'pandas', 'numpy', 'scikit-learn'
      ],
    packages=["seqprops", "seqprops.data"],
    package_dir={'seqprops': 'seqprops', 'seqprops.data': 'seqprops/data'},
    include_package_data=True,
    package_data={'seqprops.data' :['aadata.csv']}
)
