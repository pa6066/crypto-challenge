from random import randint
import Plaintext
import Chitext
import numpy as np


class Textpair:
    def __init__(self, plaintext, num, pkey):
        self.plaintext = Plaintext.Plaintext(plaintext)
        self.chitext = Chitext.Chitext(self.plaintext.generate_chitext(pkey))
        # print self.plaintext.y, self.chitext.x

