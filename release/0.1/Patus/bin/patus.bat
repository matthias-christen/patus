@echo off
setlocal enableextensions
setlocal enabledelayedexpansion

rem Get the path to the Patus home directory (this script is in the \bin directory)
set PATUS_HOME=%~dp0..

rem Show help?
set allargs=%*
if not "!allargs:--help=xxx!" == "!allargs!" (
    java -jar %PATUS_HOME%\bin\patus.jar codegen
    goto :eof
) 

rem Set default values
set arch=x86_64 SSE asm
set strategy=%PATUS_HOME%\strategy\cacheblocked.stg
set outdir=out
set unroll=1,2,4:1:1

rem Build the argument list
set args=
set first=true
set is_arch=false
set is_strategy=false
set is_outdir=false
set is_unroll=false
set is_other_arg=false
set argname=

for %%p in ( %* ) do (
    set param=%%p
    
    if "!is_arch!" == "true" (
        set arch=!param!
        set is_arch=false
    ) else if "!is_strategy!" == "true" (
        set strategy=!param!
        set is_strategy=false
    ) else if "!is_outdir!" == "true" (
        set outdir=!param!
        set is_outdir=false
    ) else if "!is_unroll!" == "true" (
        set unroll=!param!
        set is_unroll=false
    ) else if "!is_other_arg!" == "true" (
        set args=!args! !argname!=!param!
        set is_other_arg=false
    ) else (
        if "!param:~2,12!" == "architecture" (
            set is_arch=true
        ) else if "!param:~2,8!" == "strategy" (
            set is_strategy=true
        ) else if "!param:~2,6!" == "outdir" (
            set is_outdir=true
        ) else if "!param:~2,6!" == "unroll" (
            set is_unroll=true
        ) else if "!first!" == "false" (
            if "!param:~0,2!" == "--" (
                set is_other_arg=true
                set argname=!param!
            ) else (
                set args=!args! !param!
            )
        )
    )
    
    set first=false
)

rem Run Patus
java -jar %PATUS_HOME%\bin\patus.jar codegen --stencil2=%1 --strategy=%strategy% --architecture="%PATUS_HOME%\arch\architectures.xml,%arch%" --unroll=%unroll% --outdir=%outdir% %args%
endlocal

:eof