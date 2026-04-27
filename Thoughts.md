## Disclaimer
This file is supposed to be a file where I explain my thoughts during execution. Contrary to my normal way of operating, I decided to only do this section after finishing the project, whereas normally I would build it as I go. The objective was to see how well I could justify my reasoning for some decisions even after some days without being in direct contact with them. Given it's my thoughts, I'll be informal in the way I'm describing the project and sometimes I might not get straight to the point.

Since I primarily want to focus on Machine Learning, I'll just give some insights on the physics side of this problem, instead of overcomplicating things.

This also has "minimal" use of AI, 90% of it was to fetch the data, 10% was to debug some errors that I was taking way too long to understand.

Paragraphs with !!! are decisions that I changed further down the project.

Also note that the code posted on GitHub is the finished and cleaned product. There are several checks that I'll talk about in this file that are currently not in the code because they made it messy and don't influence the end result.

## Thoughts

# Dataset

First of all, the data. "I" fetched the data from the Materials Project API. Prior to finishing this work I didn't think learning how to retrieve data from an external source was particularly useful, so I just asked Claude to do it (turns out it is useful and worth understanding properly!)


!!! After looking into the materials, I decided to initially divide the dataset into metals and non-metals, since metals have overlapping valence band maximum (VBM) and conduction band minimum (CBM), meaning the band gap is 0 and there is nothing to predict (because there isn't a bandgap).

Checked for missing data, which there wasn't (should be obvious, since the values come from DFT calculatiobs so they probably would have all the parameters, but it is a good principle to check). Then I looked into the bandgap histogram, which was clearly skewed towards 0, since most of the materials are either conductors, narrow-gap semiconductors or semiconductors. There were also some "outliers", with very large band gaps, and one was clearly out of place -> H2, which it wouldn't make sense to include in the dataframe, becauase it's a gas at room temperature.

Total magnetization was clearly 0-inflated, a property that is hard for some ML models to deal, So I decided to divide them into 2 distinct features, if the property is magnetic and what is their magnetization per atom. I decided to feature engineer magnetization per atom instead of using total magnetization because the first one is an intensive properties like the bandgap, which means it doesn't scale with the size of the unit cell, while total magnetization is an extensive property, that means that magnetization per atom is more meaningfull to predict the bandgap. For tree-based models this decomposition is also convenient: the model can look at is_magnetic first, and if it's 0, ignore mag_per_atom entirely (Also defined is_magnetic as an int property so I can feed it to other models that are not tree based, if needed).

There was also clear heteroscedasticity in some scatter plots. I tried log-transforming the band gap to compress the higher values and reduce it, but it actually made things worse and less interpretable, so I kept the raw target

I proceeded to do several checks on histograms, correlation matrices and scatter plots, to look for redundant features, candidates for log-transforming and potential outliers.

!!! Since most of the features were week predictors, I decided to use Matminer's Magpie features, which include compositional descriptors like electronegativity, atomic radius, and ionicity. However, since they are pre-computed descriptors that encode the same physics I was trying to capture manually, using them felt like outsourcing the feature engineering rather than building it myself. I also had too many features to handle meaningfully (Keeping in mind that they actually didn't improve the model that much -> I lost the RMSE and R² values but I know they were pretty similar).

# Model

When building the model I first tried a simple regression model, which I already knew wasn't the best option, but I wanted the experience of implementing it.

I then implemented a tree-based model. I split the data into train and test (80/20), stratified by material class to ensure equal class representation in both sets. I defined a search space of hyperparameters (max depth, learning rate, number of estimators) and used RandomizedSearchCV over 20 iterations with 5-fold cross-validation to find the best combination. Unsurprisingly, deeper trees gave better R² and RMSE on training data, but I was seeing train R² = 0.99 and test R² = 0.94, which clearly indicated either data leakage or overfitting.

Both were true, data was leaking, because I had 2 predictors on vbm and cbm, and we can obtain directly the bandgap (Eg) with the formula Eg = cbm - vbm, which I completely forgot. Another predictor that was leaking data was formation_energy and that one I had no idea, but after some research, formation energy requires a DFT structural relaxation that is part of the same computational pipeline that produces the band gap in the Materials Project, so in practice both are computed together.

On the overfitting side, it was obvious since the model had way to many leaves. After removing the leaking features and tuning the regularization parameters, I achieved ok results (train RMSE = 0.3 eV; test RMSE = 0.7 eV; train R² = 0.94, test R² = 0.81).

For there it was a battle against reducing overfitting and improving test R² and RMSE.

Looking at predicted vs actual plots and residuals, I noticed the model was predicting insulator values for some semiconductors and vice versa near the class boundaries. To tackle this I built a classifier, first classifying each material as insulator, narrow-gap semiconductor, or semiconductor, then feeding those labels into separate regression models, one per class. The results were way worse. The classifier still got some predictions wrong near the bin edges, and since I was using the best parameters for each class separately, a misclassified material would automatically get a wrong band gap prediction.

But I liked how the pipeline looked for this purpose so I decided to keep it, even though a simpler model would grant better results, and since the focus isn't having the best model prossible but instead, to have experience implementing different solutions.

My next approach was to engineer better features, specifically, features derived from the density of states (DOS) curves that have physical correlation with the band gap: the total electronic states in the valence band near the Fermi level, the number of Van Hove singularities, the steepness of the band edge, and the mean density of states in the valence band window. The correlation matrix confirmed these were among the strongest predictors. This improved results slightly. Overfitting was still present but reduced. That data had a lot of missing data, so my approach was to fill that missing data with the mean from each material class, since deleting missing data would cause a loss of 25% of the values.

I made one final change to the model, I wanted to see how it would predict for both metals and non metals, feeding the model with the values of 0 eV in the band gap (the model that I decided to put on github), to see how it would generalize across all material types given any material. Results are the ones reported in the README.

Looking at the graphs, it is clear that the model tends to predict 0 for a lot of values, and as the band gap values go higher the model also tends to underestimate the predictions. The residuals seem well-distributed

The model has clear limitations. Given more time I would tune the tree hyperparameters further, engineer the DOS features more carefully using the true band edge position rather than a fixed energy window, and potentially revisit the Magpie compositional descriptors in a more controlled way.

The best result was achieved with the single regressor model with DOS features:

| Metric | Train | Test |
|---|---|---|
| R² | 0.942 | 0.862 |
| RMSE (eV) | 0.398 | 0.599 |
| MAE (eV) | 0.221 | 0.400 |








