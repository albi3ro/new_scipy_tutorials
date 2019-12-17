SciPy Optimize Module
=====================

Module Contents
---------------

-  Optimization

   -  Local

      -  `1D Optimization <./Optimization_1D.ipynb>`__
      -  `ND Optimization <./Optimization_ND.ipynb>`__
      -  `Linear Programming <./Linear_Prog.ipynb>`__

   -  Global Optimization

      -  `Brute <./Optimization_Global_brute.ipynb>`__
      -  `shgo <./Optimization_Global_shgo.ipynb>`__
      -  `Differential
         Evolution <./Optimization_Global_differential_evolution.ipynb>`__
      -  `Basin Hopping <./Optimization_Global_basinhopping.ipynb>`__
      -  `Dual Annealing <./Optimization_Global_dual_annealing.ipynb>`__

-  Root Finding

   -  `1D Roots <./Roots_1D.ipynb>`__
   -  `ND Roots <./Roots_ND.ipynb>`__

-  `Curve Fitting and Least Squares <./Curve_Fit.ipynb>`__

Optimization
------------

Optimization algorithms find the smallest (or greatest) value of a
function over a given range. Local methods find a point that is just the
lowest point for some neighborhood, but multiple local minima can exist
over the domain of the problem. Global methods try to find the lowest
value over an entire region.

The available local methods include ``minimize_scalar``, ``linprog``,
and ``minimize``. ``minimize_scalar`` only minimizes functions of one
variable. ``linprog`` for Linear Programming deals with only linear
functions with linear constraints. ``minimize`` is the most general
routine. It can deal with either of those subcases in addition to any
arbitrary function of many variables.

Optimize provides 5 different global optimization routines. Brute
directly computes points on a grid. This routine is the easiest to get
working, saving programmer time at the cost of computer time.
Conversely, shgo (Simplicial Homology Global Optimization) is a
conceptually difficult but powerful routine. Differential Evolution,
Basin Hopping, and Dual Annealing are all `Monte
Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__ based,
relying on random numbers. Differential Evolution is an `Evolutionary
Algorithm <https://en.wikipedia.org/wiki/Evolutionary_algorithm>`__,
creating a “population” of random points and iterating it based on which
ones give the lowest values. Basin Hopping and Dual Annealing instead
work locally through a `Markov
Chain <https://en.wikipedia.org/wiki/Markov_chain>`__ random walk. They
can computationally cope with a large number of dimensions, but they can
get locally stuck and miss the global minimum. Basin Hopping provides a
more straightforward base and options for a great deal of customization,
and dual annealing provides a more complicated method to avoid getting
locally trapped.

Root Finding
------------

```root`` <./Roots_ND.ipynb>`__ and
```root_scalar`` <./Roots_1D.ipynb>`__ solve the problem

.. math::


   f(x) = 0.

 ``root_scalar`` is limited to when :math:`x` is a scalar, while for the
more general ``root`` :math:`x` can be a multidimensional vector.

Curve Fitting
-------------

This module provides tools to optimize the fit between a parametrized
function and data. ``curve_fit`` provides a simple interface where you
don’t have to worry about the guts of fitting a function. It minimizes
the sum of the residuals over the parameters of the function, or:

.. math::


   \text{min}_{\text{p}} \quad \sum_i \big( f(x_i , \text{p} ) - y_i \big)^2.

 If you want, you can instead pass this function to the provided
non-linear least squares optimizer ``least_squares``. If the model
function is linear, ``nnls`` and ``lsq_linear`` exist as well.

Common Traits in the Submodule
------------------------------

Output: ``OptimizeResult``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr />

Many functions return an object that can contain more information than
simply “This is the minimium”. The information varies between function,
method used by the function, and flags given to function, but the way of
accessing the data remains the same.

Let’s create one of these data types via minimization to look at it:

.. code:: ipython3

    f = lambda x : x**2
    
    result=optimize.minimize(f,[2],method="BFGS")

You can determine what data types are availible via

.. code:: ipython3

    result.keys()




.. parsed-literal::

    dict_keys(['fun', 'jac', 'hess_inv', 'nfev', 'njev', 'status', 'success', 'message', 'x', 'nit'])



And you can access individual values via:

.. code:: ipython3

    result.x




.. parsed-literal::

    array([-1.88846401e-08])



Inspecting the object with ``?`` or ``??`` can tell you more about what
the individual components actually are.

In Jupyter Lab, Contextual Help, ``Ctrl+I`` can also provide this
information.

.. code:: ipython3

    ? result

``args``
~~~~~~~~

.. raw:: html

   <hr />

Many routines allow function parameters in a tuple to be passed to the
routine via the ``args`` flag:

.. code:: ipython3

    f_parameter = lambda x,a : (x-a)**2
    
    optimize.minimize(f_parameter,[0],args=(1,))




.. parsed-literal::

          fun: 5.5507662238258444e-17
     hess_inv: array([[0.5]])
          jac: array([4.68181046e-13])
      message: 'Optimization terminated successfully.'
         nfev: 9
          nit: 2
         njev: 3
       status: 0
      success: True
            x: array([0.99999999])



Methods
~~~~~~~

.. raw:: html

   <hr />

The functions in ``scipy.optimize`` are uniform wrappers that can call
to multiple different methods, algorithms, behind the scenes. For
example, ``minimize_scalar`` can use Brent, Golden, or Bounded methods.
Methods can have different strengths, weaknesses, and pitfalls. SciPy
will automatically choose certain routines given inputted information,
but if you know more about the problem, a different routine might be
better.

An example of choosing the routine:

.. code:: ipython3

    f = lambda x : x**2
    
    optimize.minimize(f,[2],method="CG")




.. parsed-literal::

         fun: 5.53784654790294e-15
         jac: array([-1.33932256e-07])
     message: 'Optimization terminated successfully.'
        nfev: 9
         nit: 1
        njev: 3
      status: 0
     success: True
           x: array([-7.44167088e-08])



Method Options
~~~~~~~~~~~~~~

.. raw:: html

   <hr />

``minimize`` itself has 14 different methods, and it’s not the only
routine that calls multiple methods. While much of the information and
functionality is unified across the routine, each method does have it’s
individual settings. The settings can be found through the
``show_options`` function:

.. code:: ipython3

    optimize.show_options(solver="minimize",method="CG")


.. parsed-literal::

    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.
    
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.


The settings are passed in a dictionary to the solver:

.. code:: ipython3

    options_dictionary = {
        "maxiter": 5,
        "eps": 1e-6
    }
    
    optimize.minimize(f,[2],options=options_dictionary)




.. parsed-literal::

          fun: 2.4972794524898593e-13
     hess_inv: array([[0.5]])
          jac: array([5.4425761e-10])
      message: 'Optimization terminated successfully.'
         nfev: 9
          nit: 2
         njev: 3
       status: 0
      success: True
            x: array([-4.99727871e-07])



Tolerance and Iterations
~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <hr />

How much computer time do you want to spend on this problem? How
accurate do you need your answer? Is your function really expensive to
calculate?

When the two successive values are within the tolerance range of each
other or the routine has reached the maximum number of iterations, the
routine will exit. Some functions differentiate between relative
tolerance and absolute tolerance. Relative tolerance scales for the
aboslute size of the values. For example, if two steps are five apart,
but each about a trillion, the function can exit. Tolerance in the
domain ``x`` direction also differs from the tolerance in the range
``f`` direction. For minimization, the ``gtol`` tolerance can also apply
to zeroing the gradient.

Some methods also allow for specifying both the maximum number of
iterations and the maximum number of function evaluations. Some methods
evaulate a function multiple times during each iteration.

Whether these quantities exist, and the procedure for setting these
quantities varies between functions and methods within functions. Check
individual documentation for details, but here is one example:

.. code:: ipython3

    optimize.minimize(f,[2],tol=1e-10,options={"maxiter":10})




.. parsed-literal::

          fun: 5.55111515902901e-17
     hess_inv: array([[0.5]])
          jac: array([-4.81884952e-17])
      message: 'Optimization terminated successfully.'
         nfev: 12
          nit: 3
         njev: 4
       status: 0
      success: True
            x: array([-7.45058062e-09])



