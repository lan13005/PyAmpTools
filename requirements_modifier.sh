# These two lines could cause problems during installation
#   program will ask to log into github to pull the git repos
sed -i '/git+/d' requirements.txt
sed -i '/pyamptools/d' requirements.txt