# CategoricalOneHotEncoder
This is a very simple extension of the scikit-learn transformer class for one-hot encoding of string categorical values in a single step without needing to (mis)use a LabelEncoder.  It fits in the normal sklearn workflow, including in pipelines, and works in the couple of instances I have used it but has not been extensively tested.

#### Note:
Sklearn recently added this functionality to the built-in OneHotEncoder, so I will probably not do any further optimization, etc. to this.  But it is still useful when I need to work with environments using older versions.