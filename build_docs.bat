RMDIR "docs" /S /Q
mkdir docs
pdoc --html -f --output-dir .docs_test toleranceinterval
robocopy .docs_test\toleranceinterval\ docs\ /E
