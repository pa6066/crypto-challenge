import numpy as np


class Chitext:
    def __init__(self, chitext):
        # self.x = self.convert_chitext(chitext)
        self.x = chitext

    def convert_chitext(self, chitext):
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

    def x_(self, index):
        """
        Access for the chitext so we can use x_(1) to get the first value in x x[0]
        :param index:
        :return:
        """
        return self.x[index - 1]

    def get_x(self, var):
        """
        Reads in a string like x_1 and returns the value of x[1 - 1]
        :param var:
        :return:
        """
        value = var.split('_')[1]
        return self.x_(int(value))


    def decrypt(self, pkey):
        pass