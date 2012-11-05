# bash shell script to set the Patus environment variables

# shallow test to see if we are in the correct directory
# Just probe to see if we have a few essential subdirectories --
# indicating that we are probably in a Patus root directory.
if [ -d "bin" ] && [ -d "arch" ] && [ -d "strategy" ] && [ -f "bin/patus.jar" ]
then
	MYPATH=`./util/fixpath "$PATH" :`
	if [ -z "$MYPATH" ]
	then
		echo "Error running ./util/fixpath"
	else
		export PATUS_HOME=$PWD
		echo "Setting PATUS_HOME to $PATUS_HOME"

		export PATH="$MYPATH":"$PATUS_HOME"/bin
		echo "Updating PATH to include $PATUS_HOME/bin"
	fi
else
	echo "Error: patusvars.sh must be sourced from within the Patus root directory"
fi
