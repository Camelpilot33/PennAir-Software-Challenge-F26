from setuptools import find_packages, setup

package_name = "pennair_app"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/pennair_app", ["pennair_app/input_p5.mp4"]),
        (
            "share/" + package_name + "/launch",
            ["launch/part5.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="samuel",
    maintainer_email="samuel@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "video_node = pennair_app.video_node:main",
            "detector_node = pennair_app.detector_node:main",
        ],
    },
)
