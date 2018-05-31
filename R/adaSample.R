#' Implementation of AdaSampling for positive unlabelled and label noise
#' learning.
#'
#' \code{adaSample()} applies the AdaSampling procedure to reduce noise
#' in the training set, and subsequently trains a classifier from
#' the new training set. For each row (observation) in the test set, it
#' returns the probabilities of it being a positive ("P) or negative
#' ("N") instance, as a two column data frame.
#'
#' \code{adaSample()} is an adaptive sampling-based noise reduction method
#' to deal with noisy class labelled data, which acts as a wrapper for
#' traditional classifiers, such as support vector machines,
#' k-nearest neighbours, logistic regression, and linear discriminant
#' analysis.
#'
#' This process is used to build up a noise-minimized training set
#' that is derived by iteratively resampling the training set,
#' (\code{train}) based on probabilities derived after its classification.
#'
#' This sampled training set is then used to train a classifier, which
#' is then executed on the test set. \code{adaSample()} returns a series of
#' predictions for each row of the test set.
#'
#' Note that this function does not evaluate the quality of the model
#' and thus does not compare its output to true values of the test set.
#' To assess please see \code{adaSvmBenchmark()}.
#'
#' @section References:
#' Yang, P., Liu, W., Yang. J. (2017) Positive unlabeled learning via wrapper-based
#' adaptive sampling. \emph{International Joint Conferences on Artificial Intelligence (IJCAI)}, 3272-3279
#'
#' Yang, P., Ormerod, J., Liu, W., Ma, C., Zomaya, A., Yang, J.(2018) 
#' AdaSampling for positive-unlabeled and label noise learning with bioinformatics applications. 
#' \emph{IEEE Transactions on Cybernetics}, doi:10.1109/TCYB.2018.2816984
#'
#' @param Ps names (name as index) of positive examples
#' @param Ns names (name as index) of negative examples
#' @param train.mat training data matrix, without class labels.
#' @param test.mat test data matrix, without class labels.
#' @param classifier classification algorithm to be used for learning. Current options are
#' support vector machine, \code{"svm"}, k-nearest neighbour, \code{"knn"}, logistic regression \code{"logit"}, or
#' linear discriminant analysis \code{"lda"}.
#' @param s sets the seed.
#' @param C sets how many times to run the classifier, C>1 induces an ensemble learning model.
#' @param sampleFactor provides a control on the sample size for resampling.
#' @return a two column matrix providing classification probabilities of each sample 
#' with respect to positive and negative classes
#' @export

adaSample <- function(Ps, Ns, train.mat, test.mat, classifier="svm", s=1, C=1, sampleFactor=1) {

  # checking the input
  if(ncol(train.mat) != ncol(test.mat)) {stop("train.mat and test.mat do not have the same number of columns")}

  # initialize sampling probablity
  pos.probs <- rep(1, length(Ps))
  una.probs <- rep(1, length(Ns))
  names(pos.probs) <- Ps
  names(una.probs) <- Ns
  
  i <- 0
  while (i < 5) {
    # update count
    i <- i + 1
    # training the predictive model
    model <- singleIter(Ps=Ps, Ns=Ns, dat=train.mat, pos.probs=pos.probs,
                 una.probs=una.probs, seed=i, classifier=classifier, sampleFactor=sampleFactor)

    # update probability arrays
    pos.probs <- model[Ps, "P"]
    una.probs <- model[Ns, "N"]
  }

  pred <- singleIter(Ps=Ps, Ns=Ns, dat=train.mat, test=test.mat,
              pos.probs=pos.probs, una.probs=una.probs, seed=s, classifier=classifier, sampleFactor=sampleFactor)

  # if C is greater than 1, create an ensemble
  if (C > 1){
    for (j in 2:C){
      pred <- pred + singleIter(Ps=Ps, Ns=Ns, dat=train.mat, test=test.mat,
                pos.probs=pos.probs, una.probs=una.probs, seed=j, classifier=classifier, sampleFactor=sampleFactor)
    }
    pred <- pred/C
  }

  return(pred)
}
