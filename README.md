# Cvx-SparsePCA
Principal Component Analysis (PCA) is a dimensionality reduction and data analysis tool commonly used in
many areas. The main idea of PCA is to represent high dimensional data with a few representative components
that capture most of the variance present in the data. However, there is an obvious disadvantage of traditional
PCA when it is applied to analyze data where interpretability is important. In applications, where the features
have some physical meanings, we lose the ability to interpret the principal components extracted by
conventional PCA because each principal component is a linear combination of all the original features. For
this reason, sparse PCA has been proposed to improve the interpretability of traditional PCA by introducing
sparsity to the loading vectors of principal components. The sparse PCA can be formulated as an ‘1 regularized
optimization problem, which can be solved by proximal gradient methods. However, these methods do not
scale well because computation of the exact gradient is generally required at each iteration. Stochastic gradient
framework addresses this challenge by computing an expected gradient at each iteration. Nevertheless,
stochastic approaches typically have low convergence rates due to the high variance. We propose
a convex sparse principal component analysis (Cvx-SPCA), which leverages a proximal variance reduced
stochastic scheme to achieve a geometric convergence rate.
