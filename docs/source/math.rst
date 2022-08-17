
Mathematical Framework for latentcor
====================================

Main Framework
--------------

**Latent Gaussian Copula Model for Mixed Data**

:code:`latentcor` utilizes the powerful semi-parametric latent Gaussian copula models to estimate
latent correlations between mixed data types (continuous/binary/ternary/truncated or zero-inflated).
Below we review the definitions for each type.

.. jupyter-execute::

    from latentcor import gen_data, latentcor

*Definition of continuous model*

A random :math:`X\in\cal{R}^{p}` satisfies the Gaussian copula (or nonparanormal) model if there
exist monotonically increasing :math:`f=(f_{j})_{j=1}^{p}` with :math:`Z_{j}=f_{j}(X_{j})` satisfying
:math:`Z\sim N_{p}(0, \Sigma)`, :math:`\sigma_{jj}=1`; we denote :math:`X\sim NPN(0, \Sigma, f)`
:cite:p:`fan2017high`.

.. jupyter-execute::

    print(gen_data(n = 6, tps = "con")['X'])

*Definition of binary model*

A random :math:`X\in\cal{R}^{p}` satisfies the binary latent Gaussian copula model if there exists :math:`W\sim NPN(0, \Sigma, f)` such that :math:`X_{j}=I(W_{j}>c_{j})`, where :math:`I(\cdot)` is the indicator function and :math:`c_{j}` are constants :cite:p:`fan2017high`.

.. jupyter-execute::

    print(gen_data(n = 6, tps = "bin")['X'])

*Definition of ternary model*

A random :math:`X\in\cal{R}^{p}` satisfies the ternary latent Gaussian copula model if there exists :math:`W\sim NPN(0, \Sigma, f)` such that :math:`X_{j}=I(W_{j}>c_{j})+I(W_{j}>c'_{j})`, where :math:`I(\cdot)` is the indicator function and :math:`c_{j}<c'_{j}` are constants :cite:p:`quan2018rank`.

.. jupyter-execute::

    print(gen_data(n = 6, tps = "ter")['X'])

*Definition of truncated or zero-inflated model*

A random :math:`X\in\cal{R}^{p}` satisfies the truncated latent Gaussian copula model if there exists :math:`W\sim NPN(0, \Sigma, f)` such that :math:`X_{j}=I(W_{j}>c_{j})W_{j}`, where :math:`I(\cdot)` is the indicator function and :math:`c_{j}` are constants :cite:p:`yoon2020sparse`.

.. jupyter-execute::

    print(gen_data(n = 6, tps = "tru")['X'])

*Mixed latent Gaussian copula model*

The mixed latent Gaussian copula model jointly models :math:`W=(W_{1}, W_{2}, W_{3}, W_{4})\sim NPN(0, \Sigma, f)` such that :math:`X_{1j}=W_{1j}`, :math:`X_{2j}=I(W_{2j}>c_{2j})`, :math:`X_{3j}=I(W_{3j}>c_{3j})+I(W_{3j}>c'_{3j})` and :math:`X_{4j}=I(W_{4j}>c_{4j})W_{4j}`.

.. jupyter-execute::

    X = gen_data(n = 100, tps = ["con", "bin", "ter", "tru"])['X']
    print(X[ :6, : ])

**Moment-based estimation of latent correlation matrix based on bridge functions**

The estimation of latent correlation matrix :math:`\Sigma` is achieved via the **bridge function** :math:`F` which is defined such that :math:`E(\hat{\tau}_{jk})=F(\sigma_{jk})`, where :math:`\sigma_{jk}` is the latent correlation between variables :math:`j` and :math:`k`, and :math:`\hat{\tau}_{jk}` is the corresponding sample Kendall's :math:`\tau`. 


*Kendall's correlation*

Given observed :math:`\mathbf{x}_{j}, \mathbf{x}_{k}\in\cal{R}^{n}`,

.. math::

    \hat{\tau}_{jk}=\hat{\tau}(\mathbf{x}_{j}, \mathbf{x}_{k})=\frac{2}{n(n-1)}\sum_{1\le i<i'\le n}sign(x_{ij}-x_{i'j})sign(x_{ik}-x_{i'k}),

where :math:`n` is the sample size.

:code:`latentcor` calculates pairwise Kendall's :math:`\widehat \tau` as part of the estimation process.

.. jupyter-execute::

    K = latentcor(X, tps = ["con", "bin", "ter", "tru"])['K']
    print(K)

Using :math:`F` and :math:`\widehat \tau_{jk}`, a moment-based estimator is :math:`\hat{\sigma}_{jk}=F^{-1}(\hat{\tau}_{jk})` with the corresponding :math:`\hat{\Sigma}` being consistent for :math:`\Sigma` :cite:p:`fan2017high,quan2018rank,yoon2020sparse`. 


The explicit form of *bridge function* :math:`F` has been derived for all combinations of continuous(C)/binary(B)/ternary(N)/truncated(T) variable types, and we summarize the corresponding references. Each of this combinations is implemented in :code:`latentcor`.


Below we provide an explicit form of :math:`F` for each combination.

*Theorem (explicit form of bridge function)*

Let :math:`W_{1}\in\cal{R}^{p_{1}}`, :math:`W_{2}\in\cal{R}^{p_{2}}`, :math:`W_{3}\in\cal{R}^{p_{3}}`,
:math:`W_{4}\in\cal{R}^{p_{4}}` be such that :math:`W=(W_{1}, W_{2}, W_{3}, W_{4})\sim NPN(0, \Sigma, f)`
with :math:`p=p_{1}+p_{2}+p_{3}+p_{4}`. Let :math:`X=(X_{1}, X_{2}, X_{3}, X_{4})\in\cal{R}^{p}`
satisfy :math:`X_{j}=W_{j}` for `j=1,...,p_{1}`, :math:`X_{j}=I(W_{j}>c_{j})`
for :math:`j=p_{1}+1, ..., p_{1}+p_{2}`, :math:`X_{j}=I(W_{j}>c_{j})+I(W_{j}>c'_{j})`
for :math:`j=p_{1}+p_{2}+1, ..., p_{3}` and :math:`X_{j}=I(W_{j}>c_{j})W_{j}`
for :math:`j=p_{1}+p_{2}+p_{3}+1, ..., p` with :math:`\Delta_{j}=f(c_{j})`.
The rank-based estimator of :math:`\Sigma` based on the observed :math:`n` realizations of
:math:`X` is the matrix :math:`\mathbf{\hat{R}}` with :math:`\hat{r}_{jj}=1`,
:math:`\hat{r}_{jk}=\hat{r}_{kj}=F^{-1}(\hat{\tau}_{jk})` with block structure

.. math::

    \mathbf{\hat{R}}=\begin{pmatrix}
    F_{CC}^{-1}(\hat{\tau}) & F_{CB}^{-1}(\hat{\tau}) & F_{CN}^{-1}(\hat{\tau}) & F_{CT}^{-1}(\hat{\tau})\\
    F_{BC}^{-1}(\hat{\tau}) & F_{BB}^{-1}(\hat{\tau}) & F_{BN}^{-1}(\hat{\tau}) & F_{BT}^{-1}(\hat{\tau})\\
    F_{NC}^{-1}(\hat{\tau}) & F_{NB}^{-1}(\hat{\tau}) & F_{NN}^{-1}(\hat{\tau}) & F_{NT}^{-1}(\hat{\tau})\\
    F_{TC}^{-1}(\hat{\tau}) & F_{TB}^{-1}(\hat{\tau}) & F_{TN}^{-1}(\hat{\tau}) & F_{TT}^{-1}(\hat{\tau})
    \end{pmatrix}

.. math::
    
    F(\cdot)=\begin{cases}
    CC: & 2\sin^{-1}(r)/\pi \\
    \\
    BC: & 4\Phi_{2}(\Delta_{j},0;r/\sqrt{2})-2\Phi(\Delta_{j}) \\
    \\
    BB: & 2\{\Phi_{2}(\Delta_{j},\Delta_{k};r)-\Phi(\Delta_{j})\Phi(\Delta_{k})\}  \\
    \\
    NC: & 4\Phi_{2}(\Delta_{j}^{2},0;r/\sqrt{2})-2\Phi(\Delta_{j}^{2})+4\Phi_{3}(\Delta_{j}^{1},\Delta_{j}^{2},0;\Sigma_{3a}(r))-2\Phi(\Delta_{j}^{1})\Phi(\Delta_{j}^{2})\\
    \\
    NB: & 2\Phi_{2}(\Delta_{j}^{2},\Delta_{k},r)\{1-\Phi(\Delta_{j}^{1})\}-2\Phi(\Delta_{j}^{2})\{\Phi(\Delta_{k})-\Phi_{2}(\Delta_{j}^{1},\Delta_{k},r)\} \\
    \\
    NN: & 2\Phi_{2}(\Delta_{j}^{2},\Delta_{k}^{2};r)\Phi_{2}(-\Delta_{j}^{1},-\Delta_{k}^{1};r)-2\{\Phi(\Delta_{j}^{2})-\Phi_{2}(\Delta_{j}^{2},\Delta_{k}^{1};r)\}\{\Phi(\Delta_{k}^{2})\\
    & -\Phi_{2}(\Delta_{j}^{1},\Delta_{k}^{2};r)\} \\
    \\
    TC: & -2\Phi_{2}(-\Delta_{j},0;1/\sqrt{2})+4\Phi_{3}(-\Delta_{j},0,0;\Sigma_{3b}(r)) \\
    \\
    TB: & 2\{1-\Phi(\Delta_{j})\}\Phi(\Delta_{k})-2\Phi_{3}(-\Delta_{j},\Delta_{k},0;\Sigma_{3c}(r))-2\Phi_{3}(-\Delta_{j},\Delta_{k},0;\Sigma_{3d}(r))  \\
    \\
    TN: & -2\Phi(-\Delta_{k}^{1})\Phi(\Delta_{k}^{2}) + 2\Phi_{3}(-\Delta_{k}^{1},\Delta_{k}^{2},\Delta_{j};\Sigma_{3e}(r)) \\
    & +2\Phi_{4}(-\Delta_{k}^{1},\Delta_{k}^{2},-\Delta_{j},0;\Sigma_{4a}(r))+2\Phi_{4}(-\Delta_{k}^{1},\Delta_{k}^{2},-\Delta_{j},0;\Sigma_{4b}(r)) \\
    \\
    TT: & -2\Phi_{4}(-\Delta_{j},-\Delta_{k},0,0;\Sigma_{4c}(r))+2\Phi_{4}(-\Delta_{j},-\Delta_{k},0,0;\Sigma_{4d}(r)) \\
    \end{cases}


where :math:`\Delta_{j}=\Phi^{-1}(\pi_{0j})`, :math:`\Delta_{k}=\Phi^{-1}(\pi_{0k})`,
:math:`\Delta_{j}^{1}=\Phi^{-1}(\pi_{0j})`, :math:`\Delta_{j}^{2}=\Phi^{-1}(\pi_{0j}+\pi_{1j})`,
:math:`\Delta_{k}^{1}=\Phi^{-1}(\pi_{0k})`, :math:`\Delta_{k}^{2}=\Phi^{-1}(\pi_{0k}+\pi_{1k})`,

.. math::

    \Sigma_{3a}(r)=
    \begin{pmatrix}
    1 & 0 & \frac{r}{\sqrt{2}} \\
    0 & 1 & -\frac{r}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & -\frac{r}{\sqrt{2}} & 1
    \end{pmatrix}, \;\;\;
    \Sigma_{3b}(r)=
    \begin{pmatrix}
    1 & \frac{1}{\sqrt{2}} & \frac{r}{\sqrt{2}}\\
    \frac{1}{\sqrt{2}} & 1 & r \\
    \frac{r}{\sqrt{2}} & r & 1
    \end{pmatrix},

.. math::

    \Sigma_{3c}(r)=
    \begin{pmatrix}
    1 & -r & \frac{1}{\sqrt{2}} \\
    -r & 1 & -\frac{r}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} & -\frac{r}{\sqrt{2}} & 1
    \end{pmatrix}, \;\;\;
    \Sigma_{3d}(r)=
    \begin{pmatrix}
    1 & 0 & -\frac{1}{\sqrt{2}} \\
    0 & 1 & -\frac{r}{\sqrt{2}} \\
    -\frac{1}{\sqrt{2}} & -\frac{r}{\sqrt{2}} & 1
    \end{pmatrix},

.. math::

    \Sigma_{3e}(r)=
    \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & r \\
    0 & r & 1
    \end{pmatrix},  \;\;\;
    \Sigma_{4a}(r)=
    \begin{pmatrix}
    1 & 0 & 0 & \frac{r}{\sqrt{2}} \\
    0 & 1 & -r & \frac{r}{\sqrt{2}} \\
    0 & -r & 1 & -\frac{1}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & \frac{r}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 1
    \end{pmatrix},

.. math::

    \Sigma_{4b}(r)=
    \begin{pmatrix}
    1 & 0 & r & \frac{r}{\sqrt{2}} \\
    0 & 1 & 0 & \frac{r}{\sqrt{2}} \\
    r & 0 & 1 & \frac{1}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & \frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 1
    \end{pmatrix}, \;\;\;
    \Sigma_{4c}(r)=
    \begin{pmatrix}
    1 & 0 & \frac{1}{\sqrt{2}} & -\frac{r}{\sqrt{2}} \\
    0 & 1 & -\frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} & -\frac{r}{\sqrt{2}} & 1 & -r \\
    -\frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} & -r & 1
    \end{pmatrix}

and

.. math::

    \Sigma_{4d}(r)=
    \begin{pmatrix}
    1 & r & \frac{1}{\sqrt{2}} & \frac{r}{\sqrt{2}} \\
    r & 1 & \frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
    \frac{1}{\sqrt{2}} & \frac{r}{\sqrt{2}} & 1 & r \\
    \frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} & r & 1
    \end{pmatrix}.


**Estimation methods**

Given the form of bridge function :math:`F`, obtaining a moment-based estimation
:math:`\widehat \sigma_{jk}` requires inversion of :math:`F`. :code:`latentcor`
implements two methods for calculation of the inversion:

* :code:`method = "original"`
* :code:`method = "approx"`
  
Both methods calculate inverse bridge function applied to each element of sample Kendall's
:math:`\tau` matrix. Because the calculation is performed point-wise (separately for each pair
of variables), the resulting point-wise estimator of correlation matrix may not be positive
semi-definite. :code:`latentcor` performs projection of the pointwise-estimator to the space of
positive semi-definite matrices, and allows for shrinkage towards identity matrix using the parameter
:code:`nu`.

*Original method (`method = "original"`)*

Original estimation approach relies on numerical inversion of :math:`F` based on solving
uni-root optimization problem. Given the calculated :math:`\widehat \tau_{jk}`
(sample Kendall's :math:`\tau` between variables :math:`j` and :math:`k`), the estimate of
latent correlation :math:`\widehat \sigma_{jk}` is obtained by calling :code:`scipy.optimize.fminbound`
function to solve the following optimization problem:

.. math::

    \widehat r_{jk} = \arg\min_{r} \{F(r) - \widehat \tau_{jk}\}^2.

The parameter :code:`tol` controls the desired accuracy of the minimizer and is passed to
:code:`scipy.optimize.fminbound`, with the default precision of :math:`10^{-8}`.

.. jupyter-execute::

    estimate_original = latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "original", tol = 1e-8)

*Algorithm for Original method*

*Input*: :math:`F(r)=F(r, \mathbf{\Delta})` - bridge function based on the type of variables :math:`j`, :math:`k`

* *Step 1*. Calculate :math:`\hat{\tau}_{jk}` using :math:`(1)`.

.. jupyter-execute::
   
    print(estimate_original['K'])
   
* *Step 2*. For binary/truncated variable :math:`j`, set :math:`\hat{\mathbf{\Delta}}_{j}=\hat{\Delta}_{j}=\Phi^{-1}(\pi_{0j})` with :math:`\pi_{0j}=\sum_{i=1}^{n}\frac{I(x_{ij}=0)}{n}`. For ternary variable :math:`j`, set :math:`\hat{\mathbf{\Delta}}_{j}=(\hat{\Delta}_{j}^{1}, \hat{\Delta}_{j}^{2})` where :math:`\hat{\Delta}_{j}^{1}=\Phi^{-1}(\pi_{0j})` and :math:`\hat{\Delta}_{j}^{2}=\Phi^{-1}(\pi_{0j}+\pi_{1j})` with :math:`\pi_{0j}=\sum_{i=1}^{n}\frac{I(x_{ij}=0)}{n}` and :math:`\pi_{1j}=\sum_{i=1}^{n}\frac{I(x_{ij}=1)}{n}`.

.. jupyter-execute::
   
    print(estimate_original['zratios'])

* *Step 3* Compute :math:`F^{-1}(\hat{\tau}_{jk})` as :math:`\hat{r}_{jk}=argmin\{F(r)-\hat{\tau}_{jk}\}^{2}` solved via :code:`scipy.optimize.fminbound` function with accuracy :code:`tol`.

.. jupyter-execute::

    print(estimate_original['Rpointwise'])

*Approximation method (`method = "approx"`)*

A faster approximation method is based on multi-linear interpolation of pre-computed inverse
bridge function on a fixed grid of points :cite:p:`yoon2021fast`. This is possible as the inverse
bridge function is an analytic function of at most :math:`5` parameters:

* Kendall's :math:`\tau`
* Proportion of zeros in the :math:`1st` variable 
* (Possibly) proportion of zeros and ones in the :math:`1st` variable
* (Possibly) proportion of zeros in the :math:`2nd` variable
* (Possibly) proportion of zeros and ones in the :math:`2nd` variable


In short, d-dimensional multi-linear interpolation uses a weighted average of :math:`2^{d}`
neighbors to approximate the function values at the points within the d-dimensional cube of
the neighbors, and to perform interpolation, :code:`latentcor` takes advantage of the :code:`Python`
package :code:`scipy.interpolate.RegularGridInterpolator`. This approximation method has been first
described in :cite:p:`yoon2021fast` for continuous/binary/truncated cases. In :code:`latentcor`,
we additionally implement ternary case, and optimize the choice of grid as well as interpolation
boundary for faster computations with smaller memory footprint.

.. jupyter-execute::

    estimate_approx = latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "approx")
    print(estimate_approx['Rpointwise'])

*Algorithm for Approximation method*

*Input*: Let :math:`\check{g}=h(g)`, pre-computed values :math:`F^{-1}(h^{-1}(\check{g}))` on a fixed grid :math:`\check{g}\in\check{\cal{G}}` based on the type of variables :math:`j` and :math:`k`. For binary/continuous case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j})`; for binary/binary case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j}, \check{\Delta}_{k})`; for truncated/continuous case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j})`; for truncated/truncated case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j}, \check{\Delta}_{k})`; for ternary/continuous case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j}^{1}, \check{\Delta}_{j}^{2})`; for ternary/binary case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j}^{1}, \check{\Delta}_{j}^{2}, \check{\Delta}_{k})`; for ternary/truncated case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j}^{1}, \check{\Delta}_{j}^{2}, \check{\Delta}_{k})`; for ternay/ternary case, :math:`\check{g}=(\check{\tau}_{jk}, \check{\Delta}_{j}^{1}, \check{\Delta}_{j}^{2}, \check{\Delta}_{k}^{1}, \check{\Delta}_{k}^{2})`.

* *Step 1* and *Step 2* same as Original method.
  
* *Step 3*. If :math:`|\hat{\tau}_{jk}|\le \mbox{ratio}\times \bar{\tau}_{jk}(\cdot)`, apply interpolation; otherwise apply Original method.

To avoid interpolation in areas with high approximation errors close to the boundary, we use hybrid scheme in *Step 3*. The parameter :code:`ratio` controls the size of the region where the interpolation is performed (:code:`ratio = 0` means no interpolation, :code:`ratio = 1` means interpolation is always performed). For the derivation of approximate bound for BC, BB, TC, TB, TT cases see @yoon2021fast. The derivation of approximate bound for NC, NB, NN, NT case is in the Appendix.

.. math::

    \bar{\tau}_{jk}(\cdot)=
    \begin{cases}
    2\pi_{0j}(1-\pi_{0j})  &   for \; BC \; case\\
    2\min(\pi_{0j},\pi_{0k})\{1-\max(\pi_{0j}, \pi_{0k})\}  &   for \; BB \; case\\
    2\{\pi_{0j}(1-\pi_{0j})+\pi_{1j}(1-\pi_{0j}-\pi_{1j})\}  &   for \; NC \; case\\
    2\min(\pi_{0j}(1-\pi_{0j})+\pi_{1j}(1-\pi_{0j}-\pi_{1j}),\pi_{0k}(1-\pi_{0k}))  &   for \; NB \; case\\
    2\min(\pi_{0j}(1-\pi_{0j})+\pi_{1j}(1-\pi_{0j}-\pi_{1j}), \\
    \;\;\;\;\;\;\;\;\;\;\pi_{0k}(1-\pi_{0k})+\pi_{1k}(1-\pi_{0k}-\pi_{1k}))  &   for \; NN \; case\\
    1-(\pi_{0j})^{2}  &   for \; TC \; case\\
    2\max(\pi_{0k},1-\pi_{0k})\{1-\max(\pi_{0k},1-\pi_{0k},\pi_{0j})\}  &   for \; TB \; case\\
    1-\{\max(\pi_{0j},\pi_{0k},\pi_{1k},1-\pi_{0k}-\pi_{1k})\}^{2}  &   for \; TN \; case\\
    1-\{\max(\pi_{0j},\pi_{0k})\}^{2}  &   for \; TT \; case\\
    \end{cases}

By default, :code:`latentcor` uses :code:`ratio = 0.9` as this value was recommended in @yoon2021fast having a good balance of accuracy and computational speed. This value, however, can be modified by the user

.. jupyter-execute::

    print(latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "approx", ratio = 0.99)['R'])
    print(latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "approx", ratio = 0.4)['R'])
    print(latentcor(X, tps = ["con", "bin", "ter", "tru"], method = "original")['R'])

The lower is the :code:`ratio`, the closer is the approximation method to original method
(with :code:`ratio = 0` being equivalent to :code:`method = "original"`), but also the higher
is the cost of computations.

*Rescaled Grid for Interpolation*

Since :math:`|\hat{\tau}|\le \bar{\tau}`, the grid does not need to cover the whole domain
:math:`\tau\in[-1, 1]`. To optimize memory associated with storing the grid, we rescale :math:`\tau`
as follows:

.. math::

    \check{\tau}_{jk}=\tau_{jk}/\bar{\tau}_{jk}\in[-1, 1],

where :math:`\bar{\tau}_{jk}` is as defined above. 

In addition, for ternary variable :math:`j`, it always holds that

.. math::

    \Delta_{j}^{2}>\Delta_{j}^{1}` since :math:`\Delta_{j}^{1}=\Phi^{-1}(\pi_{0j})

and

.. math::
    
    \Delta_{j}^{2}=\Phi^{-1}(\pi_{0j}+\pi_{1j}).
    
Thus, the grid should not cover the the area corresponding to

.. math::
    
    \Delta_{j}^{2}\ge\Delta_{j}^{1}.
    
We thus rescale as follows:

.. math::
    
    \check{\Delta}_{j}^{1}=\Delta_{j}^{1}/\Delta_{j}^{2}\in[0, 1];
    
.. math::
    
    \check{\Delta}_{j}^{2}=\Delta_{j}^{2}\in[0, 1].

**Adjustment of pointwise-estimator for positive-definiteness**

Since the estimation is performed point-wise, the resulting matrix of estimated latent correlations
is not guaranteed to be positive semi-definite. For example, this could be expected when the sample
size is small (and so the estimation error for each pairwise correlation is larger).

.. jupyter-execute::

    X = gen_data(n = 6, tps = ["con", "bin", "ter", "tru"])['X']
    print(latentcor(X, tps = ["con", "bin", "ter", "tru"])['Rpointwise'])

:code:`latentcor` automatically corrects the pointwise estimator to be positive definite by making
two adjustments.
First, if :code:`Rpointwise` has smallest eigenvalue less than zero, the :code:`latentcor` projects
this matrix to
the nearest positive semi-definite matrix.
The user is notified of this adjustment through the message (supressed in previous code chunk), e.g.

.. jupyter-execute::

    print(latentcor(X, tps = ["con", "bin", "ter", "tru"])['R'])

Second, :code:`latentcor` shrinks the adjusted matrix of correlations towards identity matrix using
the parameter :code:`\nu` with default value of 0.001 (:code:`nu = 0.001`), so that the resulting
:code:`latentcor[0]` is strictly positive definite with the minimal eigenvalue being greater or equal
to :code:`\nu`. That is

.. math::

    R = (1 - \nu) \widetilde R + \nu I,

where :code:`\widetilde R` is the nearest positive semi-definite matrix to :code:`Rpointwise`.

.. jupyter-execute::

    print(latentcor(X, tps = ["con", "bin", "ter", "tru"], nu = 0.001)['R'])

As a result, :code:`R` and :code:`Rpointwise` could be quite different when sample size :code:`n`
is small. When :code:`n` is large and :code:`p` is moderate, the difference is typically driven by
parameter :code:`nu`.

.. jupyter-execute::

    X = gen_data(n = 100, tps = ["con", "bin", "ter", "tru"])['X']
    out = latentcor(X, tps = ["con", "bin", "ter", "tru"], nu = 0.001)
    print(out['Rpointwise'])
    print(out['R'])

Appendix
--------

*Derivation of bridge function for ternary/truncated case*

Without loss of generality, let :math:`j=1` and :math:`k=2`. By the definition of Kendall's :math:`\tau`,

.. math::

    \tau_{12}=E(\hat{\tau}_{12})=E[\frac{2}{n(n-1)}\sum_{1\leq i\leq i' \leq n} sign\{(X_{i1}-X_{i'1})(X_{i2}-X_{i'2})\}].

Since :math:`X_{1}` is ternary,

.. math::

    \begin{align}
    &sign(X_{1}-X_{1}') \nonumber\\ =&[I(U_{1}>C_{11},U_{1}'\leq C_{11})+I(U_{1}>C_{12},U_{1}'\leq C_{12})-I(U_{1}>C_{12},U_{1}'\leq C_{11})] \nonumber\\
    &-[I(U_{1}\leq C_{11}, U_{1}'>C_{11})+I(U_{1}\leq C_{12}, U_{1}'>C_{12})-I(U_{1}\leq C_{11}, U_{1}'>C_{12})] \nonumber\\
    =&[I(U_{1}>C_{11})-I(U_{1}>C_{11},U_{1}'>C_{11})+I(U_{1}>C_{12})-I(U_{1}>C_{12},U_{1}'>C_{12}) \nonumber\\
    &-I(U_{1}>C_{12})+I(U_{1}>C_{12},U_{1}'>C_{11})] \nonumber\\
    &-[I(U_{1}'>C_{11})-I(U_{1}>C_{11},U_{1}'>C_{11})+I(U_{1}'>C_{12})-I(U_{1}>C_{12},U_{1}'>C_{12}) \nonumber\\
    &-I(U_{1}'>C_{12})+I(U_{1}>C_{11},U_{1}'>C_{12})] \nonumber\\
    =&I(U_{1}>C_{11})+I(U_{1}>C_{12},U_{1}'>C_{11})-I(U_{1}'>C_{11})-I(U_{1}>C_{11},U_{1}'>C_{12}) \nonumber\\
    =&I(U_{1}>C_{11},U_{1}'\leq C_{12})-I(U_{1}'>C_{11},U_{1}\leq C_{12}).
    \end{align}

Since :math:`X_{2}` is truncated, :math:`C_{1}>0` and

.. math::

    \begin{align}
    sign(X_{2}-X_{2}')=&-I(X_{2}=0,X_{2}'>0)+I(X_{2}>0,X_{2}'=0) \nonumber\\
    &+I(X_{2}>0,X_{2}'>0)sign(X_{2}-X_{2}') \nonumber\\
    =&-I(X_{2}=0)+I(X_{2}'=0)+I(X_{2}>0,X_{2}'>0)sign(X_{2}-X_{2}').
    \end{align}

Since :math:`f` is monotonically increasing, :math:`sign(X_{2}-X_{2}')=sign(Z_{2}-Z_{2}')`,

.. math::

    \begin{align}
    \tau_{12}=&E[I(U_{1}>C_{11},U_{1}'\leq C_{12}) sign(X_{2}-X_{2}')] \nonumber\\ &-E[I(U_{1}'>C_{11},U_{1}\leq C_{12}) sign(X_{2}-X_{2}')] \nonumber\\
    =&-E[I(U_{1}>C_{11},U_{1}'\leq C_{12}) I(X_{2}=0)] \nonumber\\
    &+E[I(U_{1}>C_{11},U_{1}'\leq C_{12}) I(X_{2}'=0)] \nonumber\\
    &+E[I(U_{1}>C_{11},U_{1}'\leq C_{12})I(X_{2}>0,X_{2}'>0)sign(Z_{2}-Z_{2}')] \nonumber\\
    &+E[I(U_{1}'>C_{11},U_{1}\leq C_{12}) I(X_{2}=0)] \nonumber\\
    &-E[I(U_{1}'>C_{11},U_{1}\leq C_{12}) I(X_{2}'=0)] \nonumber\\
    &-E[I(U_{1}'>C_{11},U_{1}\leq C_{12})I(X_{2}>0,X_{2}'>0)sign(Z_{2}-Z_{2}')]  \nonumber\\
    =&-2E[I(U_{1}>C_{11},U_{1}'\leq C_{12}) I(X_{2}=0)] \nonumber\\
    &+2E[I(U_{1}>C_{11},U_{1}'\leq C_{12}) I(X_{2}'=0)] \nonumber\\
    &+E[I(U_{1}>C_{11},U_{1}'\leq C_{12})I(X_{2}>0,X_{2}'>0)sign(Z_{2}-Z_{2}')] \nonumber\\
    &-E[I(U_{1}'>C_{11},U_{1}\leq C_{12})I(X_{2}>0,X_{2}'>0)sign(Z_{2}-Z_{2}')].
    \end{align}

From the definition of :math:`U`, let :math:`Z_{j}=f_{j}(U_{j})` and :math:`\Delta_{j}=f_{j}(C_{j})` for :math:`j=1,2`. Using :math:`sign(x)=2I(x>0)-1`, we obtain

.. math::

    \begin{align}
    \tau_{12}=&-2E[I(Z_{1}>\Delta_{11},Z_{1}'\leq \Delta_{12},Z_{2}\leq \Delta_{2})]+2E[I(Z_{1}>\Delta_{11},Z_{1}'\leq \Delta_{12},Z_{2}'\leq \Delta_{2})] \nonumber\\
    &+2E[I(Z_{1}>\Delta_{11},Z_{1}'\leq \Delta_{12})I(Z_{2}>\Delta_{2},Z_{2}'>\Delta_{2},Z_{2}-Z_{2}'>0)] \nonumber\\
    &-2E[I(Z_{1}'>\Delta_{11},Z_{1}\leq \Delta_{12})I(Z_{2}>\Delta_{2},Z_{2}'>\Delta_{2},Z_{2}-Z_{2}'>0)] \nonumber\\
    =&-2E[I(Z_{1}>\Delta_{11},Z_{1}'\leq \Delta_{12}, Z_{2}\leq \Delta_{2})]+2E[I(Z_{1}>\Delta_{11},Z_{1}'\leq \Delta_{12}, Z_{2}'\leq \Delta_{2})] \nonumber\\
    &+2E[I(Z_{1}>\Delta_{11},Z_{1}'\leq\Delta_{12},Z_{2}'>\Delta_{2},Z_{2}>Z_{2}')] \nonumber\\
    &-2E[I(Z_{1}'>\Delta_{11},Z_{1}\leq\Delta_{12},Z_{2}'>\Delta_{2},Z_{2}>Z_{2}')].
    \end{align}

Since :math:`\{\frac{Z_{2}'-Z_{2}}{\sqrt{2}}, -Z{1}\}`, :math:`\{\frac{Z_{2}'-Z_{2}}{\sqrt{2}}, Z{1}'\}` and :math:`\{\frac{Z_{2}'-Z_{2}}{\sqrt{2}}, -Z{2}'\}` are standard bivariate normally distributed variables with correlation :math:`-\frac{1}{\sqrt{2}}$, $r/\sqrt{2}` and :math:`-\frac{r}{\sqrt{2}}`, respectively, by the definition of :math:`\Phi_3(\cdot,\cdot, \cdot;\cdot)` and :math:`\Phi_4(\cdot,\cdot, \cdot,\cdot;\cdot)` we have

.. math::

    \begin{align}
    F_{NT}(r;\Delta_{j}^{1},\Delta_{j}^{2},\Delta_{k})= & -2\Phi_{3}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},\Delta_{k};\begin{pmatrix}
    1 & 0 & -r \\
    0 & 1 & 0 \\
    -r & 0 & 1
    \end{pmatrix} \right\} \nonumber\\
    &+2\Phi_{3}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},\Delta_{k};\begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & r \\
    0 & r & 1
    \end{pmatrix}\right\}\nonumber \\
    & +2\Phi_{4}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k},0;\begin{pmatrix}
    1 & 0 & 0 & \frac{r}{\sqrt{2}} \\
    0 & 1 & -r & \frac{r}{\sqrt{2}} \\
    0 & -r & 1 & -\frac{1}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & \frac{r}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 1
    \end{pmatrix}\right\} \nonumber\\
    &-2\Phi_{4}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k},0;\begin{pmatrix}
    1 & 0 & r & -\frac{r}{\sqrt{2}} \\
    0 & 1 & 0 & -\frac{r}{\sqrt{2}} \\
    r & 0 & 1 & -\frac{1}{\sqrt{2}} \\
    -\frac{r}{\sqrt{2}} & -\frac{r}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 1
    \end{pmatrix}\right\}.
    \end{align}

Using the facts that

.. math::

    \begin{align}
    &\Phi_{4}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k},0;\begin{pmatrix}
    1 & 0 & r & -\frac{r}{\sqrt{2}} \\
    0 & 1 & 0 & -\frac{r}{\sqrt{2}} \\
    r & 0 & 1 & -\frac{1}{\sqrt{2}} \\
    -\frac{r}{\sqrt{2}} & -\frac{r}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 1
    \end{pmatrix}\right\} \nonumber\\ &+\Phi_{4}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k},0;\begin{pmatrix}
    1 & 0 & r & \frac{r}{\sqrt{2}} \\
    0 & 1 & 0 & \frac{r}{\sqrt{2}} \\
    r & 0 & 1 & \frac{1}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & \frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 1
    \end{pmatrix}\right\} \nonumber\\
    =&\Phi_{3}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k};\begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & r \\
    0 & r & 1
    \end{pmatrix}\right\}
    \end{align}

and

.. math::

    \begin{align}
    &\Phi_{3}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k};\begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & r \\
    0 & r & 1
    \end{pmatrix}\right\}+\Phi_{3}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},\Delta_{k};\begin{pmatrix}
    1 & 0 & -r \\
    0 & 1 & 0 \\
    -r & 0 & 1
    \end{pmatrix} \right\} \nonumber\\
    =&\Phi_{2}(-\Delta_{j}^{1},\Delta_{j}^{2};0)
    =\Phi(-\Delta_{j}^{1})\Phi(\Delta_{j}^{2}).
    \end{align}

So that,

.. math::

    \begin{align}
    F_{NT}(r;\Delta_{j}^{1},\Delta_{j}^{2},\Delta_{k})= & -2\Phi(-\Delta_{j}^{1})\Phi(\Delta_{j}^{2}) \nonumber\\
    &+2\Phi_{3}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},\Delta_{k};\begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & r \\
    0 & r & 1
    \end{pmatrix}\right\}\nonumber \\
    & +2\Phi_{4}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k},0;\begin{pmatrix}
    1 & 0 & 0 & \frac{r}{\sqrt{2}} \\
    0 & 1 & -r & \frac{r}{\sqrt{2}} \\
    0 & -r & 1 & -\frac{1}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & \frac{r}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 1
    \end{pmatrix}\right\} \nonumber\\
    &+2\Phi_{4}\left\{-\Delta_{j}^{1},\Delta_{j}^{2},-\Delta_{k},0;\begin{pmatrix}
    1 & 0 & r & \frac{r}{\sqrt{2}} \\
    0 & 1 & 0 & \frac{r}{\sqrt{2}} \\
    r & 0 & 1 & \frac{1}{\sqrt{2}} \\
    \frac{r}{\sqrt{2}} & \frac{r}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 1
    \end{pmatrix}\right\}.
    \end{align}

It is easy to get the bridge function for truncated/ternary case by switching :math:`j` and :math:`k`.

*Derivation of approximate bound for the ternary/continuous case*

Let :math:`n_{0x}=\sum_{i=1}^{n_x}I(x_{i}=0)`, :math:`n_{2x}=\sum_{i=1}^{n_x}I(x_{i}=2)`, :math:`\pi_{0x}=\frac{n_{0x}}{n_{x}}` and :math:`\pi_{2x}=\frac{n_{2x}}{n_{x}}`, then

.. math::

    \begin{align}
    |\tau(\mathbf{x})|\leq & \frac{n_{0x}(n-n_{0x})+n_{2x}(n-n_{0x}-n_{2x})}{\begin{pmatrix} n \\ 2 \end{pmatrix}} \nonumber\\
    = & 2\{\frac{n_{0x}}{n-1}-(\frac{n_{0x}}{n})(\frac{n_{0x}}{n-1})+\frac{n_{2x}}{n-1}-(\frac{n_{2x}}{n})(\frac{n_{0x}}{n-1})-(\frac{n_{2x}}{n})(\frac{n_{2x}}{n-1})\} \nonumber\\
    \approx & 2\{\frac{n_{0x}}{n}-(\frac{n_{0x}}{n})^2+\frac{n_{2x}}{n}-(\frac{n_{2x}}{n})(\frac{n_{0x}}{n})-(\frac{n_{2x}}{n})^2\} \nonumber\\
    = & 2\{\pi_{0x}(1-\pi_{0x})+\pi_{2x}(1-\pi_{0x}-\pi_{2x})\}
    \end{align}

For ternary/binary and ternary/ternary cases, we combine the two individual bounds.


*Derivation of approximate bound for the ternary/truncated case*

Let :math:`\mathbf{x}\in\mathcal{R}^{n}` and :math:`\mathbf{y}\in\mathcal{R}^{n}` be the observed :math:`n` realizations of ternary and truncated variables, respectively. Let :math:`n_{0x}=\sum_{i=0}^{n}I(x_{i}=0)`, :math:`\pi_{0x}=\frac{n_{0x}}{n}`, :math:`n_{1x}=\sum_{i=0}^{n}I(x_{i}=1)`, :math:`\pi_{1x}=\frac{n_{1x}}{n}`, :math:`n_{2x}=\sum_{i=0}^{n}I(x_{i}=2)`, :math:`\pi_{2x}=\frac{n_{2x}}{n}`,
:math:`n_{0y}=\sum_{i=0}^{n}I(y_{i}=0)`, :math:`\pi_{0y}=\frac{n_{0y}}{n}`, :math:`n_{0x0y}=\sum_{i=0}^{n}I(x_{i}=0 \;\& \; y_{i}=0)`, :math:`n_{1x0y}=\sum_{i=0}^{n}I(x_{i}=1 \;\& \; y_{i}=0)` and
:math:`n_{2x0y}=\sum_{i=0}^{n}I(x_{i}=2 \;\& \; y_{i}=0)` then

.. math::

    \begin{align}
    |\tau(\mathbf{x}, \mathbf{y})|\leq &
    \frac{\begin{pmatrix}n \\ 2\end{pmatrix}-\begin{pmatrix}n_{0x} \\ 2\end{pmatrix}-\begin{pmatrix}n_{1x} \\ 2\end{pmatrix}-\begin{pmatrix} n_{2x} \\ 2 \end{pmatrix}-\begin{pmatrix}n_{0y} \\ 2\end{pmatrix}+\begin{pmatrix}n_{0x0y} \\ 2 \end{pmatrix}+\begin{pmatrix}n_{1x0y} \\ 2\end{pmatrix}+\begin{pmatrix}n_{2x0y} \\ 2\end{pmatrix}}{\begin{pmatrix}n \\ 2\end{pmatrix}} \nonumber
    \end{align}

Since :math:`n_{0x0y}\leq\min(n_{0x},n_{0y})`, :math:`n_{1x0y}\leq\min(n_{1x},n_{0y})` and :math:`n_{2x0y}\leq\min(n_{2x},n_{0y})` we obtain

.. math::

    \begin{align}
    |\tau(\mathbf{x}, \mathbf{y})|\leq &
    \frac{\begin{pmatrix}n \\ 2\end{pmatrix}-\begin{pmatrix}n_{0x} \\ 2\end{pmatrix}-\begin{pmatrix}n_{1x} \\ 2\end{pmatrix}-\begin{pmatrix} n_{2x} \\ 2 \end{pmatrix}-\begin{pmatrix}n_{0y} \\ 2\end{pmatrix}}{\begin{pmatrix}n \\ 2\end{pmatrix}} \nonumber\\
    & +  \frac{\begin{pmatrix}\min(n_{0x},n_{0y}) \\ 2 \end{pmatrix}+\begin{pmatrix}\min(n_{1x},n_{0y}) \\ 2\end{pmatrix}+\begin{pmatrix}\min(n_{2x},n_{0y}) \\ 2\end{pmatrix}}{\begin{pmatrix}n \\ 2\end{pmatrix}} \nonumber\\
    \leq & \frac{\begin{pmatrix}n \\ 2\end{pmatrix}-\begin{pmatrix}\max(n_{0x},n_{1x},n_{2x},n_{0y}) \\ 2\end{pmatrix}}{\begin{pmatrix}n \\ 2\end{pmatrix}} \nonumber\\
    \leq & 1-\frac{\max(n_{0x},n_{1x},n_{2x},n_{0y})(\max(n_{0x},n_{1x},n_{2x},n_{0y})-1)}{n(n-1)} \nonumber\\
    \approx & 1-(\frac{\max(n_{0x},n_{1x},n_{2x},n_{0y})}{n})^{2} \nonumber\\
    =& 1-\{\max(\pi_{0x},\pi_{1x},\pi_{2x},\pi_{0y})\}^{2} \nonumber\\
    =& 1-\{\max(\pi_{0x},(1-\pi_{0x}-\pi_{2x}),\pi_{2x},\pi_{0y})\}^{2}
    \end{align}

It is easy to get the approximate bound for truncated/ternary case by switching :math:`\mathbf{x}` and :math:`\mathbf{y}`.
