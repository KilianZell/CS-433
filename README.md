# EPFL_ML_project_1
Contributors: Krieger Benjamin, Zell Kilian, Jegou Sam

Competition: Higgs Boson Machine Learning Challenge

## Summary
Our best model was a ridge regression, with a polynomial augmentation of degree 5 and some other feature augmentations.

## Features Selection
All the outliers were set to the mean of their corresponding features, before the standardization process (0 mean, 1 standard deviation).

It has been tried to remove the features that had too much outliers (~90%), or separate the dataset in function of the indicator variable JET, but it decreased our accuracy.

## The model
Our model is a simple ridge regression, with regularization of $\lambda = 10^{-5}$.
The input data has been augmented via:
  - a polynomial augmentation of degree 5
  - some other augmentations for the first features, with cosines, sinus, sinc, and others to add non linearity

Then we project the prediction on {-1,1} with a threshold at $0.05$.

## Generating a solution
Put the files `run.ipynb` and `implementations.py` in the main folder.

In the same folder, create a folder 'Data' containing the data file 'sample-submission.csv'.

Run the file `run.py`. It will generate in your current folder a file 'submission.csv', containing the prediction of our model.

## References
Machine learning course at EPFL:

Description of the dataset: https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf
