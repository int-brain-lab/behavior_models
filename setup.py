from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]

setup(
    name='behavior_models',
    version='0.1.0',
    author='Charles Findling',
    url='https://github.com/csmfindling/behavior_models',
    long_description=long_description,
    install_requires=require,
)
