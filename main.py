import numpy as np

class Matrix:
    def __init__(self, n, m, rows):
        self.n = n
        self.m = m
        self.rows = rows

    def __str__(self):
        return '\n'.join(' '.join(str(val) for val in row) for row in self.rows)

    def __mul__(self, other):
        if self.m != other.n:
            raise Exception("Matrix multiplication error: dimension mismatch.")
        result = []
        for i in range(self.n):
            new_row = []
            for j in range(other.m):
                val = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.m))
                new_row.append(val)
            result.append(new_row)
        return Matrix(self.n, other.m, result)

    def __add__(self, other):
        if self.n != other.n or self.m != other.m:
            raise Exception("Matrix addition error: dimension mismatch.")
        result = [
            [self.rows[i][j] + other.rows[i][j] for j in range(self.m)]
            for i in range(self.n)
        ]
        return Matrix(self.n, self.m, result)

    def __rmul__(self, scalar):
        return self.scalar_mul(scalar)

    @staticmethod
    def identity(n):
        return Matrix(n, n, [[1 if i == j else 0 for j in range(n)] for i in range(n)])

    def copy(self):
        return Matrix(self.n, self.m, [row[:] for row in self.rows])

    def scalar_mul(self, c):
        return Matrix(self.n, self.m, [[c * val for val in row] for row in self.rows])

    def power_binary(self, k):
        if self.n != self.m:
            raise Exception("Matrix must be square.")
        if k == 0:
            return Matrix.identity(self.n)
        if k == 1:
            return self
        result = Matrix.identity(self.n)
        base = self.copy()
        exp = k
        while exp > 0:
            if exp % 2 == 1:
                result = result * base
            base = base * base
            exp //= 2
        return result

def cayley_hamilton_power(A, k):
    if A.n != A.m:
        raise Exception("Cayley-Hamilton method requires a square matrix.")

    arr = np.array(A.rows, dtype=float)
    eigenvalues, _ = np.linalg.eig(arr)
    coeffs = np.poly(eigenvalues)

    n = A.n
    powers = [None] * max(k + 1, n)
    powers[0] = Matrix.identity(n)
    for i in range(1, n):
        powers[i] = powers[i - 1] * A

    if k < n:
        return powers[k]

    for i in range(n, k + 1):
        Ai = Matrix(n, n, [[0] * n for _ in range(n)])
        for j in range(1, n + 1):
            Ai = Ai + powers[i - j].scalar_mul(coeffs[j])
        Ai = Ai.scalar_mul(-1)
        powers[i] = Ai

    return powers[k]


if __name__ == "__main__":
    N, M = map(int, input("행렬의 행과 열의 개수를 입력하세요 (공백 구분): ").split())
    K = int(input("거듭제곱할 지수를 입력하세요: "))
    A_data = []
    for i in range(N):
        row = list(map(int, input(f"{i + 1}행 입력 (공백 구분): ").split()))
        if len(row) != M:
            raise Exception("행렬의 열 수가 올바르지 않습니다.")
        A_data.append(row)

    A = Matrix(N, M, A_data)

    A_pow_bin = A.power_binary(K)
    print(f"\nA^{K} (Binary Exponentiation):")
    print(A_pow_bin)
    print()

    if N == M:
        A_pow_ch = cayley_hamilton_power(A, K)
        print(f"A^{K} (Cayley-Hamilton):")
        print(A_pow_ch)
        print()

        diff = [[float(A_pow_bin.rows[i][j] - A_pow_ch.rows[i][j]) for j in range(M)] for i in range(N)]
        print("Difference:")
        for row in diff:
            print(' '.join(map(str, row)))
    else:
        print("Cayley-Hamilton method requires a square matrix. Skipping.")