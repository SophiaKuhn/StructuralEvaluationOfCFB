from setuptools import setup, find_packages

dependencies = [
    'compas_fea',
    'websockets',
    'requests'
]

setup(
    name='strucenglib_connect',
    version='0.2',
    author="StrucEngLib Authors",
    description="",
    install_requires=dependencies,
    url="https://github.com/kfmResearch-NumericsTeam/Struc_Eng_Library_Server",
    include_package_data=True,
    packages=['strucenglib_connect', 'strucenglib_connect.server', 'strucenglib_connect.server.assets'],
    # packages=find_packages(exclude=['ez_setup', 'tests', 'tests.*']),
    entry_points={
        'console_scripts': ['strucenglib_server=strucenglib_connect.server.wsserver:main'],
    }
)
