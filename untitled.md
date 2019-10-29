One Dimensional Minimization

While nd minimization methods can work in one dimension, the multidimensional alternative works as well.  Three different options are availible, but one is almost uniformly better than the others.  Bounds or brackets can be provided.  


Pondering on iteration count: 

What to say about maxiter: it can be changed.  Should controls for max

Should I talk about tolerances and iterations in the introduction?  will be a fairly common theme across modules...


ND minimization

Functions differ in time to compute one function evaluation, number of dimensions, degree of non-linearity, constraints, differentiability, and pre-conditioning.  These, except for conditioning, affect the choice of optimization routine. For example, if a function takes a long time to compute, you want a method that makes the best of each evaluation.  Before passing a function to a routine, you can choose to precondition the function to make the optimization more efficient.

If a function does not have derivatives, then the Nelder-Mead method can perform optimization.  All the other methods use either approximated or exact derivatives.

The majority of optimization routines work by locally approximating the function. These optimizations include an analytically or numerically calculated gradient, and some also include the 2nd deriative hessian as well.
