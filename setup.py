from setuptools import setup

package_name = 'guide_llm'

setup(
    name=package_name,
    version='2.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'anthropic>=0.25.0',
    ],
    zip_safe=True,
    maintainer='Sangmim',
    maintainer_email='sangmim@student.uts.edu.au',
    description='LLM navigation agent using scene graph + Claude tool_use + Nav2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'scene_graph_agent = scene_graph_agent:main',
            'scene_graph_publisher = scene_graph_publisher:main',
        ],
    },
)
