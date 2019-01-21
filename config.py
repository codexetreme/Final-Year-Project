import os

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    #Batch Size: The number of documents that it will take per minibatch,
    # Default: 64
    BATCH_SIZE = 64

    # Total words used in the vocab list, higher the number, slower will be the training.
    VOCAB_SIZE = 6500

    # The maximum total number of words per sentence. There may be padding to make sure every
    # sentence is of uniform 50 words length, this is defined by the USE_PADDING_FOR_SENTENCE option
    MAX_WORDS_PER_SENTENCE = 50

    # Sets wether the sentences have to be padded or not.
    # True (default): will pad sentences to MAX_WORDS_PER_SENTENCE length,
    # False: no padding is applied
    USE_PADDING_FOR_SENTENCE = True

    # String that represents the padding.
    PADDING_SEQ='<PAD>'

    # Maximum number of sentences per document, these may be padded,
    # defined by USE_PADDING_FOR_DOCUMENT
    MAX_SENTENCES_PER_DOCUMENT = 50


    # This is the minimum number of times the word should occur in the corpus to be included in the 
    # vocabulary list
    MIN_VOCAB_COUNT = 5

    # The dimentions for the words from glove. Default = 100
    WORD_DIMENTIONS = 100
    # Sets wether the sentences have to be padded or not.
    # True: will pad documents to MAX_SENTENCES_PER_DOCUMENT length,
    # False (default): no padding is applied
    USE_PADDING_FOR_DOCUMENT = False


    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small




    class Paths(object):
        """Class to hold all the path config variables"""
        
        # Set this to a compiled version of glove (ie the build folder)
        GLOVE_PATH = ''

        # Set this is to the path of the text file that contains the dataset corpus
        PATH_TO_CORPUS = ''

        # This path is where the VOCAB_FILE_NAME file will be saved
        PATH_TO_VOCAB_TXT = ''
        
        # Path to the root folder of processed pkl and .dat files for glove
        PROCESSED_GLOVE_PATH = ''
        
        # The path to the dataset
        PATH_TO_DATASET = ''

        # Path to folder for dataset's selected vocabulary
        # This contains the word2id and the vectors as .dat files
        PATH_TO_DATASET_S_VOCAB = ''


        def __init__(self):
            # Config.Paths.PATH_TO_VOCAB_TXT = os.path.join(Config.Paths.PATH_TO_VOCAB_TXT,Config.Paths.VOCAB_FILE_NAME)
            # assert GLOVE_PATH is not None,"GLOVE_PATH is not SET, please set it to the build folder of glove(v1.2)" 
            # assert PATH_TO_CORPUS is not None,"PATH_TO_CORPUS is not SET, please set it to the dataset's corpus text file" 
            # assert PATH_TO_VOCAB_TXT is not None,"PATH_TO_VOCAB_TXT is not SET, please set it to a location which is acccesible by you" 
            pass
    def __init__(self,make_vocab=False):
        """Set values of computed attributes."""
        # Effective batch size

        self.paths = Config.Paths()
        pass
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
