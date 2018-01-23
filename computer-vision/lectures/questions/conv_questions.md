corrrelation vs convolution filters:
same if kernel is symmetric
correlation => exact pattern matching of repetition detection
convolution => fourrier transform (minus sign haas good properties) # frequecy domain


rk about complexity:
complexity of a linear kernel based conv filter: N^2\*W^2
complexity of a median filter N\*O(w^2 (log(W^2)))

preserve edges in a filter:
 - tends to blur edges if all weights in the kernel are positive
 - negative weights act as high pass filter 
