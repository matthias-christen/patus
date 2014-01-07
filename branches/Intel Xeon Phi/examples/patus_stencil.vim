" Vim syntax file
" Language: Patus / Stencil Specification
" Maintainer: Matthias Christen
" Latest Revision: 22 August 2012

if exists("b:current_syntax")
  finish
endif

" Keywords
syn keyword ssKeywords stencil domainsize operation boundaries initial grid param
syn match ssKeywords "iterate\s\+while"
syn keyword ssTypes float double
"syn match ssOperator "[\+\-\*\/\^\<\>\=]"
syn region ssComment start="/\*" end="\*/"
syn region ssCommentL start="//" skip="\\$" end="$" keepend

syn match ssNumber "π"
"integer number, or floating point number without a dot and with "f".
syn case ignore
syn match cNumbers display transparent "\<\d\|\.\d" contains=cNumber,cFloat
syn match cNumber display contained "\d\+\(u\=l\{0,2}\|ll\=u\)\>"
syn match cFloat display contained "\d\+f"
"floating point number, with dot, optional exponent
syn match cFloat display contained "\d\+\.\d\+\(e[-+]\=\d\+\)\=[fl]\="
"floating point number, starting with a dot, optional exponent
syn match cFloat display contained "\.\d\+\(e[-+]\=\d\+\)\=[fl]\=\>"
" Avoid highlighting '..'
syn match cNone display "\.\{2}"
"floating point number, without dot, with exponent
syn match cFloat display contained "\d\+e[-+]\=\d\+[fl]\=\>"

syn region ssSubscript start="\[" end="\]"


let b:current_syntax = "patus-stencil"

hi def link ssKeywords Statement
hi def link ssTypes Type
hi def link ssOperator Operator
hi def link ssComment Comment
hi def link ssCommentL Comment
hi def link ssNumber Number
hi def link cNumber Number
hi def link cFloat Number
hi def link ssSubscript Structure
