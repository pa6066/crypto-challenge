import numpy as np
from operator import xor, mul


class Plaintext:
    def __init__(self, plaintext):
        # self.y = self.convert_plaintext(plaintext)
        self.y = plaintext

    def convert_plaintext(self, chitext):
        """
        Convert the chitext so the contents are ints
        :param chitext:
        :return:
        """
        return np.asarray(map(int, self.split_file(chitext)))

    @staticmethod
    def split_file(keyfile):
        """Remove unnecessary characters from the chitext"""
        return keyfile.replace('[', '').replace(']', '').replace(' ', '').replace('\n', '').replace('\r', '').split(',')

    def y_(self, index):
        """
        Access for the chitext so we can use x_(1) to get the first value in x x[0]
        :param index:
        :return:
        """
        return self.y[index - 1]

    def get_y(self, var):
        """
        Reads in a string like x_1 and returns the value of x[1 - 1]
        :param var:
        :return:
        """
        value = var.split('_')[1]
        return self.y_(int(value))

    def mult_items(self, *args):
        # print(list(args))
        args = map(lambda y: self.get_y(y), list(args)[0])
        return reduce(lambda x, y: mul(x, y), args)

    def generate_chitext(self, pkey):
        """
        Encryption of given plaintext
        :param pkey:
        :return: chitext for the given plaintext
        """
        chitext = []
        for row in pkey.key:
            elems_to_add = []
            for mult_elem in row.split('+'):  # For every pair that has to be multiplied
                elems = mult_elem.split('*')
                elems_to_add.append(self.mult_items(elems))  # multiply the elems like x_1*x_2
            chitext.append(
                reduce(lambda x, y: xor(x, y), elems_to_add))  # Add the different pairs together like x_1*x_2 + x_3*x_4
        return chitext
