from AudioNetwork import AudioNetwork

def train(epochs=3, validate=True, autosave=False):
    if validate not in [True, "True", False, "False"]:
        raise ValueError("'validate' should be True or False")

    if autosave not in [True, "True", False, "False"]:
        raise ValueError("'autosave' should be True or False")

    #from serpent.machine_learning.context_classification.context_classifier import ContextClassifier
    #ContextClassifier.executable_train(epochs=int(epochs), validate=argv_is_true(validate), autosave=argv_is_true(autosave))

    AudioNetwork.executable_train(epochs=int(epochs), validate=validate, autosave=autosave)

if __name__ == '__main__':
    train(3,True,False)