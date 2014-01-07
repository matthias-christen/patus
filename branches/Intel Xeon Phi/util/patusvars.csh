# csh/tcsh shell script to set the Patus environment variables

# shallow test to see if we are in the correct directory
# Just probe to see if we have a few essential subdirectories --
# indicating that we are probably in a Patus root directory.
if ( ! -d "bin" || ! -d "arch" || ! -d "strategy" || ! -f "bin/patus.jar" ) then
	echo "Error: patusvars.csh must be sourced from within the Patus root directory"
	exit
endif

set MYPATH = `./util/fixpath "$PATH" :`
if ( $MYPATH == "" ) then
	echo "Error running ./util/fixpath"
	exit
endif

setenv PATUS_HOME "$cwd"
echo "Setting PATUS_HOME to $PATUS_HOME"

setenv PATH "$MYPATH":"$PATUS_HOME/bin"
echo "Updating PATH to include $PATUS_HOME/bin"
