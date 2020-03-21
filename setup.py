import os
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand
from setuptools.command.sdist import sdist as SdistCommand

from setuptools.command.install import install

# Force the wheel to be platform specific
# https://stackoverflow.com/a/45150383/3549270
# There's also the much more concise solution in
# https://stackoverflow.com/a/53463910/3549270,
# but that would requires python-dev
try:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False


except ImportError:
    bdist_wheel = None


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        source_dir = os.path.dirname(os.path.abspath(__file__))
        executable_name = "maturin.exe" if sys.platform.startswith("win") else "maturin"

        # Shortcut for development
        existing_binary = os.path.join(source_dir, "target", "debug", executable_name)
        if os.path.isfile(existing_binary):
            source = existing_binary
        else:
            if not shutil.which("cargo"):
                raise RuntimeError(
                    "cargo not found in PATH. Please install rust "
                    "(https://www.rust-lang.org/tools/install) and try again"
                )
            subprocess.check_call(
                ["cargo", "rustc", "--bin", "maturin", "--", "-C", "link-arg=-s"]
            )
            source = os.path.join(source_dir, "target", "debug", executable_name)
        # run this after trying to build with cargo (as otherwise this leaves
        # venv in a bad state: https://github.com/benfred/py-spy/issues/69)
        install.run(self)

        target = os.path.join(self.install_scripts, executable_name)
        os.makedirs(self.install_scripts, exist_ok=True)
        self.copy_file(source, target)
        self.copy_tree(
            os.path.join(source_dir, "maturin"),
            os.path.join(self.root or self.install_lib, "maturin"),
        )


# class CargoModifiedSdist(SdistCommand):
    # """Modifies Cargo.toml to use an absolute rather than a relative path

    # The current implementation of PEP 517 in pip always does builds in an
    # isolated temporary directory. This causes problems with the build, because
    # Cargo.toml necessarily refers to the current version of pyo3 by a relative
    # path.

    # Since these sdists are never meant to be used for anything other than
    # tox / pip installs, at sdist build time, we will modify the Cargo.toml
    # in the sdist archive to include an *absolute* path to pyo3.
    # """

    # def make_release_tree(self, base_dir, files):
        # """Stages the files to be included in archives"""
        # super().make_release_tree(base_dir, files)

        # import toml

        # # Cargo.toml is now staged and ready to be modified
        # cargo_loc = os.path.join(base_dir, "Cargo.toml")
        # assert os.path.exists(cargo_loc)

        # with open(cargo_loc, "r") as f:
            # cargo_toml = toml.load(f)

        # rel_pyo3_path = cargo_toml["dependencies"]["pyo3"]["path"]
        # base_path = os.path.dirname(__file__)
        # abs_pyo3_path = os.path.abspath(os.path.join(base_path, rel_pyo3_path))

        # cargo_toml["dependencies"]["pyo3"]["path"] = abs_pyo3_path

        # with open(cargo_loc, "w") as f:
            # toml.dump(cargo_toml, f)


# class PyTest(TestCommand):
    # user_options = []

    # def run(self):
        # self.run_command("test_rust")

        # import subprocess

        # subprocess.check_call(["pytest", "tests"])

def make_rust_extensions():
    from setuptools_rust import RustExtension
    return [RustExtension("rs_fec_conv.rs_fec_conv", "Cargo.toml", debug=True)]


setup_requires = ["setuptools-rust>=0.10.1", "wheel"]
install_requires = []
tests_require = install_requires + ["pytest", "pytest-benchmark"]

setup(
    name="rs_fec_conv",
    version="0.4.0",
	author="Benjamin Roepken",
	author_email="broepken57@hotmail.com",
	url="https://github.com/grayfox57/rs_fec_conv",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Software Development :: Build Tools",
		"Programming Language :: Rust",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=["rs_fec_conv"],
    rust_extensions=make_rust_extensions(),
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False,
    #cmdclass={"install": PostInstallCommand, "bdist_wheel": bdist_wheel, "test": PyTest, "sdist": CargoModifiedSdist},
	#cmdclass={"install": PostInstallCommand, "bdist_wheel": bdist_wheel},
	cmdclass={"install": PostInstallCommand, "test": PyTest, "sdist": CargoModifiedSdist},
)