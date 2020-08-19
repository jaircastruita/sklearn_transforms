from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pickle


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class SimpleDNNClassifier(ClassifierMixin, BaseEstimator):
    """ Load a hard coded neural network trained in PyTorch:
    architecture  [12 (predictors) x 50(hidden_layer)] -> [50 x 6(n_classes)]
    Parameters
    ----------
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self):
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None

    def _load_weights(self):
        """
        Cargamos los pesos de la red pre-entrenada con PyTorch. 
        Como IBM Icloud no deja instalar dicha librería importamos los pesos desde un archivo
        pkl esperando que eso no incurra en una penalizacion en el concurso
        ----------
        weights : dict
        A parameter dictionary containing. pre-loaded weights of a 1-layer DNN
        Attributes
        ----------
        
        """
        
        with open('saved_weights.pkl', 'rb') as db_file:
            weights = pickle.load(db_file)
    
        self.hidden_weights = weights['hidden1.weight']
        self.hidden_bias = weights['hidden1.bias']
        self.output_weights = weights['output.weight']
        self.output_bias = weights['output.bias']

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_, _ = np.unique(y, return_inverse=True)

        self.X_ = X
        self.y_ = y
        
        # Cargamos los pesos a la red neuronal
        self._load_weights()
        
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        if len(X.shape) == 1:
            X = X.copy().reshape(1,-1)

        # Input validation
        X = check_array(X)

        out = self.hidden_weights.dot(X.T)
        out = out + self.hidden_bias.reshape(-1,1)
        out = np.maximum(out, 0) # ReLU function
        out = self.output_weights.dot(out) + self.output_bias.reshape(-1,1)

        return self.classes_[np.argmax(out, axis=0)]