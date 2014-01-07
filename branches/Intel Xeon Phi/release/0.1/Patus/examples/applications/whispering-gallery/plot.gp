unset key
set terminal pdfcairo
set output "output.pdf"
set view map
set size square
set nocbtics
set title "Time Averaged Electromagnetic Energy u" 
set xrange [ -0.5 : @PLOTSIZE@ ] noreverse nowriteback
set yrange [ -0.5 : @PLOTSIZE@ ] noreverse nowriteback
set cblabel "u_em" 
#set cbrange [ 0 : 1700 ] noreverse nowriteback
set palette rgbformulae 33, 13, 10
splot 'output.txt' matrix with image

