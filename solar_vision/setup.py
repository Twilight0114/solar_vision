from setuptools import find_packages, setup

package_name = 'solar_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cat',
    maintainer_email='cat@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 格式： '终端运行的命令名 = 包名.文件名:main函数'
            'image_sub_node = solar_vision.image_sub_node:main',
            'my_cam_pub = solar_vision.my_cam_pub:main',
        ],
    },
)
