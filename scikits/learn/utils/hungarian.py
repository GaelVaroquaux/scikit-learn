"""
Solve the unique lowest-cost assignment problem using the
Hungarian algorithm (also known as Munkres algorithm).

References
==========

1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html

2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.

3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.

5. http://en.wikipedia.org/wiki/Hungarian_algorithm

"""
# Based on original code by Brain Clapper, adapted to numpy to scikits
# learn coding standards by G. Varoquaux

# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, G Varoquaux
# Author: Brian M. Clapper, G Varoquaux
# LICENSE: BSD

import numpy as np

################################################################################
# Object-oriented form of the algorithm
class _Hungarian(object):
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.

        :Parameters:
            cost_matrix : list of lists
                The cost matrix. If this cost matrix is not square, it
                will be padded with zeros, via a call to ``pad_matrix()``.
                (This method does *not* modify the caller's matrix. It
                operates on a copy of the matrix.)

                **WARNING**: This code handles square and rectangular
                matrices. It does *not* handle irregular matrices.

        :rtype: list
        :return: A list of ``(row, column)`` tuples that describe the lowest
                 cost path through the matrix

        """
        self.C = cost_matrix.copy()
        self.n = n = self.C.shape[0]
        self.row_covered = np.zeros(n, dtype=np.bool)
        self.col_covered = np.zeros(n, dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((2*n, 2), dtype=int)
        self.marked = np.zeros((n, n), dtype=int)

        done = False
        step = 1

        steps = { 1 : self._step1,
                  2 : self._step2,
                  3 : self._step3,
                  4 : self._step4,
                  5 : self._step5,
                  6 : self._step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(n):
            for j in range(n):
                if self.marked[i, j] == 1:
                    results += [(i, j)]

        return results

    def _step1(self):
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        self.C -= self.C.min(axis=1)[:, np.newaxis]
        return 2

    def _step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i, j] == 0) and \
                                (not self.col_covered[j]) and \
                                (not self.row_covered[i]):
                    self.marked[i, j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True

        self._clear_covers()
        return 3

    def _step3(self):
        """
        Cover each column containing a starred zero. If n columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i, j] == 1:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7 # done
        else:
            step = 4

        return step

    def _step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            row, col = self._find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row, col] = 2
                star_col = self._find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def _step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count, 0] = self.Z0_r
        path[count, 1] = self.Z0_c
        done = False
        while not done:
            row = self._find_star_in_col(path[count, 1])
            if row >= 0:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count-1, 1]
            else:
                done = True

            if not done:
                col = self._find_prime_in_row(path[count, 0])
                count += 1
                path[count, 0] = path[count-1, 0]
                path[count, 1] = col

        self._convert_path(path, count)
        self._clear_covers()
        # Erase all prime markings
        self.marked[self.marked == 2] = 0
        return 3

    def _step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self._find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i, j] += minval
                if not self.col_covered[j]:
                    self.C[i, j] -= minval
        return 4

    def _find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        return np.min(self.C[np.logical_not(self.col_covered)
                            *np.logical_not(self.row_covered[:, np.newaxis])])

    def _find_a_zero(self):
        """Find the first uncovered element with value 0"""
        C = ((self.C==0)*(1-self.col_covered)
                        *(1-self.row_covered[:, np.newaxis]))
        raveled_idx = np.argmax(C)
        col = raveled_idx % self.n
        row = raveled_idx // self.n
        if C[row, col] == 0:
            return -1, -1
        return row, col

    def _find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = np.argmax(self.marked[row] == 1)
        if not self.marked[row, col] == 1:
            col = -1
        return col

    def _find_star_in_col(self, col):
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = np.argmax(self.marked[:, col] == 1)
        if not self.marked[row, col] == 1:
            row = -1
        return row
 
    def _find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row, j] == 2:
                col = j
                break

        return col

    def _convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i, 0], path[i, 1]] == 1:
                self.marked[path[i, 0], path[i, 1]] = 0
            else:
                self.marked[path[i, 0], path[i, 1]] = 1

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_covered[:] = False
        self.col_covered[:] = False



################################################################################
# Functional form for easier use
def hungarian(cost_matrix):
    """ Return the indices to permute the columns of the matrix
        to minimize its trace.
    """
    H = _Hungarian()
    indices = H.compute(cost_matrix)
    indices.sort()
    return np.array(indices).T[1]


def find_permutation(vectors, reference):
    """ Returns the permutation indices of the vectors to maximize the
        correlation to the reference
    """
    # Compute a correlation matrix
    reference = reference/(reference**2).sum(axis=-1)[:, np.newaxis]
    vectors = vectors/(vectors**2).sum(axis=-1)[:, np.newaxis]
    K = np.abs(np.dot(reference, vectors.T))
    K -= 1 + K.max()
    K *= -1
    return hungarian(K)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    matrices = [
                # Square
                ([[400, 150, 400],
                  [400, 450, 600],
                  [300, 225, 300]],
                 850 # expected cost
                ),

                ## Rectangular variant
                #([[400, 150, 400, 1],
                #  [400, 450, 600, 2],
                #  [300, 225, 300, 3]],
                # 452 # expected cost
                #),

                # Square
                ([[10, 10,  8],
                  [ 9,  8,  1],
                  [ 9,  7,  4]],
                 18
                ),

                ## Rectangular variant
                #([[10, 10,  8, 11],
                #  [ 9,  8,  1, 1],
                #  [ 9,  7,  4, 10]],
                # 15
                #),
               ]

    m = _Hungarian()
    for cost_matrix, expected_total in matrices:
        print np.array(cost_matrix)
        cost_matrix = np.array(cost_matrix)
        indexes = m.compute(cost_matrix)
        total_cost = 0
        for r, c in indexes:
            x = cost_matrix[r, c]
            total_cost += x
            print '(%d, %d) -> %d' % (r, c, x)
        print 'lowest cost=%d' % total_cost
        assert expected_total == total_cost

