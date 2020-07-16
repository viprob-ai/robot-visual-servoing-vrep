from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['Avoidance_Task','Kinematics','Object_Detection','Plot_Data','Visual_Servoing'],
    package_dir={'': 'src'}
)

setup(**d)
