#!/usr/bin/env bash
rm -r docs
mkdir docs
pdoc --html -f --output-dir .docs_test toleranceinterval
cp -r .docs_test/toleranceinterval/* docs/
