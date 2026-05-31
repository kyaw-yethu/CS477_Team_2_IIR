import os
from glob import glob
from setuptools import setup

package_name = 'team_2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'llama-cpp-python',
        'google-genai',
        'Pillow',
        'numpy',
        'opencv-python-headless',
    ],
    zip_safe=True,
    maintainer='team_2',
    maintainer_email='team_2@todo.todo',
    description='CS477 IIR Picking Challenge - team 2',
    license='BSD',
    entry_points={
        'console_scripts': [
                    'instruction_parser = team_2.instruction_parser:main',
                    'perception_node = team_2.perception_node:main',
                    'orchestrator_node = team_2.orchestrator_node:main',
                    'grasp_server = team_2.grasp_server:main',
                    'motion_planner = team_2.motion_planner:main',
                ],
    },
)
