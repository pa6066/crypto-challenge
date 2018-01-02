import re
import Chitext
import PublicKey
import numpy as np
import Textpair
from math import pow, sqrt
from operator import xor, mul
import random
from random import SystemRandom
from multiprocessing.pool import Pool
# import pathos.pools as pp
import time
import json

np.set_printoptions(threshold=np.inf)


def unwrap_self_pairs(arg, **kwarg):
    """
    Helper function for multiprocessing
    :param arg:
    :param kwarg:
    :return:
    """
    return KeyReader.chitext_creation_process(*arg, **kwarg)


gen = SystemRandom()


def generate_plaintext(length):
    """
    Generieren von Klartext als liste
    :param length:
    :return:
    """
    # return [1, 0, 1, 0, 0] # For testing purposes
    # return np.asarray([randint(0, 1) for x in xrange(length)])
    return np.asarray([gen.randrange(2) for x in xrange(length)])

class KeyReader:
    """
    Anzahl Relationen 43
    """
    def __init__(self, keyfile, rel):
        self.pkey = PublicKey.PublicKey(self.read_keyfile(keyfile))
        self.rel = rel
        self.num = self.get_num()

    @staticmethod
    def read_keyfile(keyfile):
        """
        Read in the public key we are given
        :param keyfile:
        :return:
        """
        with open(keyfile, 'rb') as inputstream:
            keystring = inputstream.read()
        return keystring

    @staticmethod
    def read_chitext(chitextfile):
        """
        Read in the chitext we have to solve
        :param chitextfile:
        :return:
        """
        with open(chitextfile, 'rb') as inputstream:
            keystring = inputstream.read()
        return keystring

    def gauss(self, A):
        """Gaussian elimination in gf(2) with the generated keypairs"""
        num_columns = A.shape[1]
        num_rows = A.shape[0]
        # try:
        offset = 0
        for k_col in xrange(0, num_columns):
            k_row = k_col + offset
            # print "K: %d, %d" % (k_col, k_row)
            if k_row == num_columns:  # The rows after a pivot in the last column ha been chosen are all zero
                print "Matrix has row echelon form"
                return A
            maxindex = A[k_col:, k_row].argmax() + k_col # check where the first 1 is located in the column under the alst pivot element
            while A[maxindex, k_col + offset] == 0:  # As long as we can't find a pivot keep searching in the next column
                print "Matrix is singular."
                offset += 1  # Offset if there wasn't a pivot element in one column
                k_row = k_col + offset
                if k_row == num_columns:
                    print "Matrix has row echelon form"
                    return A
                maxindex = A[k_col:, k_row].argmax() + k_col
            if maxindex != k_col:  # Swap the columns so the pivot element is first
                A[[maxindex, k_col]] = A[[k_col, maxindex]]

            for row in range(0, num_rows):  # Xor rows which have a 1 under the pivot element
                if row == k_col:  # skip when comparing with itself
                    continue
                # print "ROW: %d" % row
                if A[row, k_col + offset] == 0:  # ignore if row doesn't start with 1
                    continue
                A[row, :] = np.logical_xor(A[row, :], A[k_col, :])
        return A

    def get_num(self):
        """
        :return: The number of coefficients we have
        """

        # print max(map(int, self.pkey.key[0].replace('+', '').replace('*', '').split('x_')[1:]))
        return max(map(int, self.pkey.key[0].replace('+', '').replace('*', '').split('x_')[1:]))

    def chitext_creation_process(self, i):
        """
        Create a keypair with given plaintext
        :param i: not used needed for multiprocessing
        :return: the created keypair
        """
        return self.create_relation(self.generate_pair(generate_plaintext(self.num)))

    def generate_pairs(self, rel):
        """
        Generate keypairs with randomly generated plaintext and the corresponding chitext
        :param rel:
        :return: list with all keypairs
        """
        print self.num
        p = Pool(4)  # number of processes running at the same time
        i = range(0, self.rel)
        relation = np.array(p.map(unwrap_self_pairs, zip([self]*len(i), i)), dtype=int) # Spawn process for every keypair we create
        p.close()
        p.join()
        return relation[:, 0, :]

    def generate_pair(self, plaintext):
        """
        generate a single keypair with given plaintext
        :param plaintext:
        :return: textpair
        """
        return Textpair.Textpair(plaintext, self.num, self.pkey)

    def create_relation(self, textpair):
        """
        multiply all elements of the plaintext and chitext together and return new relation
        :param textpair:
        :return:
        """
        relation = np.zeros((1, (len(textpair.chitext.x) * len(textpair.plaintext.y))))
        k = 0
        for i in xrange(1, len(textpair.plaintext.y) + 1):
            for j in xrange(1, len(textpair.chitext.x) + 1):
                # print "%d * %d = %d" % (textpair.chitext.x_(j), textpair.plaintext.y_(i), textpair.chitext.x_(j) * textpair.plaintext.y_(i))
                relation[0, k] = (mul(textpair.plaintext.y_(i), textpair.chitext.x_(j)))
                k += 1
        return relation

    def build_special_rel(self, A):
        """
        perform the back-substitution
        :param A:
        :return: list with all free variables and dependencies
        """
        length = int(pow(self.num, 2))
        # length = 25
        vars = ['2' for i in xrange(length)]
        for i in xrange(A.shape[0] - 1, -1, -1):
            if i == 14:
                pass
            unknowns = 0
            cur_unknowns = []
            for j in xrange(0, A.shape[1]):
                if A[i, j] == 1:
                    if vars[j] == '2':
                        unknowns += 1
                        cur_unknowns.append(j)
                    if isinstance(vars[j], list):
                        unknowns += 1
                        cur_unknowns.append(j)
            if unknowns == 0:
                continue
            if unknowns == 1:
                vars[cur_unknowns[0]] = 0
            elif unknowns > 1:
                vars[cur_unknowns[0]] = [cur_unknowns[unknown] for unknown in xrange(1, len(cur_unknowns))]
            else:
                raise ValueError("ERROR")
        return vars

    def check_inputs(self, special_rel, rel):
        """
        Create all possible combinations of free variables and insert into special relation
        :param special_rel:
        :param rel:
        :return:
        """

        def fill_in(possible_val):
            """
            Check if the solution to the possible value is 0
            :param possible_val:
            :return: return True if result = 0 else return False
            """
            for row in xrange(0, rel.shape[0]):
                if np.count_nonzero(rel[row, :]) == 0:
                    continue
                values = np.logical_and(rel[row, :], possible_val)
                solution = reduce(lambda x, y: int(xor(x, y)), values)
                # print solution
                if solution != 0:
                    print "test False"
                    return False
            print "TEST True"
            return True

        def insert(binary):
            """Insert values for free variables and calculate all variables
            """
            # binary = [1, 0, 1, 0, 1]
            rel_copy = list(special_rel)
            cur_unknown = 0
            for i in xrange(len(rel_copy) - 1, -1, -1):
                if rel_copy[i] == '2':
                    rel_copy[i] = binary[cur_unknown]
                    cur_unknown += 1
                if rel_copy == '0':
                    rel_copy[i] = 0
            for i in xrange(len(rel_copy) - 1, -1, -1):
                if isinstance(rel_copy[i], list):
                    result = rel_copy[rel_copy[i][0]]
                    if len(rel_copy[i]) > 1:
                        for j in xrange(1, len(rel_copy[i])):
                            result = xor(result, rel_copy[rel_copy[i][j]])
                    rel_copy[i] = result
            return rel_copy

        def get_bin(index, digit):
            """
            return 1 if free variable is 1 return 0 if not
            :param index:
            :param digit:
            :return:
            """
            if index >= len(digit):
                return 0
            return int(digit[len(digit) - 1 - index])

        unknowns = 0
        possible_values = []
        for entry in special_rel:
            if entry == '2':
                unknowns += 1
        print "UNKNOWNS: ", unknowns
        # max_num = int(pow(2, unknowns))
        max_num = unknowns + 1
        i = 1
        while i < max_num:
            binary = []
            binary_digit = bin(i).split('b')[1]
            for j in range(unknowns - 1, -1, -1):
                binary.append(get_bin(j, binary_digit))
            possible_value = insert(binary)
            if fill_in(possible_value):
                possible_values.append(possible_value)
            i += 1
        return possible_values

    def determine_plaintext(self,  possible_relations, chitext):
        """Insert chitext into the solved gauss_relation
            :returns a list with possible plaintexts"""
        max_xy_val = int(sqrt(possible_relations.shape[1]))
        results = []
        x_vals = []
        for row in xrange(0, possible_relations.shape[0]):
            x_val = []
            result = []

            cur_y_val = 0
            cur_x_val = 0
            temp = 0
            for elem in possible_relations[row, :]:
                if elem == 1:
                    temp = xor(temp, chitext[cur_y_val])
                    result.append(chitext[cur_y_val])

                else:
                    result.append("None")

                cur_y_val += 1
                if cur_y_val == max_xy_val:
                    x_val.append(temp)
                    temp = 0
                    cur_x_val += 1
                    cur_y_val = 0
            results.append(result)
            x_vals.append(x_val)
        return x_vals

    def start(self):
        """
        Start the Attack
        :return:
        """

        start = time.time()
        print "Creating Textpairs"
        relations = self.generate_pairs(self.rel)
        print "Creating Textpairs done."
        end = time.time()
        time_rel = end - start
        start = time.time()
        # print relations
        print "Starting Gauss elimination"
        gauss_matrix = self.gauss(relations)
        np.savetxt('c://users/Pascal/Desktop/cryptochallenge/docs/gauss.txt', gauss_matrix)
        print gauss_matrix
        tot = 0
        lala = np.count_nonzero(gauss_matrix, axis=1)
        for entry in lala:
            if entry == 0:
                tot += 1
        print "Laenge: ", len(lala)
        print "Anzahl zero rows: ", tot
        print "Gaussian elimination done."
        end = time.time()
        time_gauss = end - start
        start = time.time()
        print "Starting Back-Substitution"
        rollup = self.build_special_rel(gauss_matrix)
        with open('c://users/Pascal/Desktop/cryptochallenge/docs/free_vars.txt', 'wb') as outfile:
            outfile.write(json.dumps(rollup))
        # print rollup
        print "Back-Subnstitution done"
        print rollup
        # quit()
        end = time.time()
        time_spec = end - start
        start = time.time()
        print "Starting Brute-Forcing for free variables"
        solutions = self.check_inputs(rollup, relations)
        print "Possible solutions created"
        print len(solutions)
        end = time.time()
        time_check = end - start
        print time_rel, time_gauss, time_spec, time_check
        # quit()
        chitext = [1, 1, 1, 0, 0]
        for entry in self.determine_plaintext(np.asarray(solutions), chitext):
            textpair = Textpair.Textpair(entry, self.num, self.pkey)
            print "\nPLAINTEXT: ", entry
            print "CHITEXT: ", textpair.chitext.x
            if textpair.chitext.x == chitext:
                print "HUARRAY"
        quit()

        A_relation = np.array([
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            ], dtype=int
        )
        # A_relation = np.array([
        #     [0., 0., 1., 0., 0.],
        #     [1., 1., 0., 1., 1.],
        #     [1., 0., 0., 1., 0.],
        #     [1., 0., 1., 1., 1.],
        #     [0., 0., 1., 1., 0.],
        #     [1., 1., 1., 1., 0.],
        #     [0., 1., 1., 0., 0.],
        # ], dtype=int)
        # A_relation = np.array([
        #     [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        #     [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        # ], dtype=int)

        # solved_relation = self.gauss(A_relation)
        # print solved_relation
        # quit()
        # lala = np.count_nonzero(solved_relation, axis=1)
        # tot = 0
        # for entry in lala:
        #     if entry == 0:
        #         tot += 1
        # print "Laenge: ", len(lala)
        # print "Anzahl zero rows: ", tot
        # special_rel = self.build_special_rel(solved_relation)
        # # print special_rel
        # binaries = self.check_inputs(special_rel, A_relation)
        # # print np.asarray(binaries)
        # chitext = [1, 1, 1, 0, 0]
        # for entry in self.determine_plaintext(np.asarray(binaries), chitext):
        #     textpair = Textpair.Textpair(entry, self.num, self.pkey)
        #     print "\nPLAINTEXT: ", entry
        #     print "CHITEXT: ", textpair.chitext.x
        #     if textpair.chitext.x == chitext:
        #         print "HUARRAY"

if __name__ == '__main__':
    # reader = KeyReader('C:\Users\Pascal\Desktop/cryptochallenge\\public_key.txt', 2000) # Genug Gleichungen erzeugen, sodass die Matrix nie fehlerhaft singular sein kann.
    reader = KeyReader('C:\Users\Pascal\Desktop/cryptochallenge\\test_pkey.txt', 50) # Genug Gleichungen erzeugen, sodass die Matrix nie fehlerhaft singular sein kann.
    reader.start()