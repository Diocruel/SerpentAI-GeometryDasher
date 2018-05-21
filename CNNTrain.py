from ImageNetwork import ImageNetwork

def train(epochs=3, validate=True, autosave=False):
    if validate not in [True, "True", False, "False"]:
        raise ValueError("'validate' should be True or False")

    if autosave not in [True, "True", False, "False"]:
        raise ValueError("'autosave' should be True or False")

    #from serpent.machine_learning.context_classification.context_classifier import ContextClassifier
    #ContextClassifier.executable_train(epochs=int(epochs), validate=argv_is_true(validate), autosave=argv_is_true(autosave))

    ImageNetwork.executable_train(epochs=int(epochs), validate=validate, autosave=autosave)

if __name__ == '__main__':
    train(5,True,False)