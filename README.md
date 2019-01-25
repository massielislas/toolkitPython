# toolkitPython
Python port of [BYU CS 478 machine learning toolkit](http://axon.cs.byu.edu/~martinez/classes/478/stuff/Toolkit.html)

Works with Python 2.7 or 3. Requires [NumPy](http://www.numpy.org) and [SciPy](https://www.scipy.org/).

## Capabilities

1. Parses and stores the ARFF file
2. Randomizes the instances in the ARFF file
3. Provides different evaluation methods.
4. Parse command-line arguments
5. Normalize attributes

## Usage

In order to use this toolkit, most commands will be similar to those given
on the class website for the Java and C++ toolkits. With the assumption that
you already have NumPy installed (see their [website](http://www.numpy.org) for
installation instructions), usage is straight-forward.

As example, execute the following commands from the root directory of this
repository.

```bash
mkdir datasets
wget http://axon.cs.byu.edu/~martinez/classes/478/stuff/iris.arff -P datasets/
python -m toolkit.manager -L baseline -A datasets/iris.arff -E training
```

Notice that you must specify the module "toolkit" as well as the manager file. 
Aside from this difference, commands follow the same syntax as the other toolkits.

The toolkit is run as follows:
python toolkit.manager -L [learningAlgorithm] -A [ARFF\_File] -E [EvaluationMethod] {[ExtraParamters]} [-N] [-R seed]

Possible evaluation methods are:
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E training
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E static [TestARFF_File]
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E random [PercentageForTraining]
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]

For information on the expected syntax, run

```bash
python -m toolkit.manager
OR
python -m toolkit.manager --help
```

## Creating Learners

See the baseline_learner.py and its `BaselineLearner` class for an example of
the format of the learner. Learning models should inherit from the `SupervisedLearner` base class and override
the `train()` and `predict()` functions.
## Testing

Simple unit tests have been implemented to ensure that the toolkit is operating as expected. They can be run as follows (while inside the toolkitPython directory):
```bash
python -m toolkit.test_matrix
python -m toolkit.test_baseline_learner
```
