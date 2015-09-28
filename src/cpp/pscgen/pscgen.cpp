#include <iostream>
using namespace std;

extern "C" {
    void dgemv_(char* TRANS, const int* M, const int* N, double* alpha,
                double* A, const int* LDA, double* X, const int* INCX,
                double* beta, double* C, const int* INCY);

    int ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w,
               float *work, int *lwork, int *info);
}

int main() {
    int i,j;
    float *A = new float[16];
    A[0] = 1; A[1] = 1; A[2] = 1;  A[3] = 1;
    A[4] = 1; A[5] = 2; A[6] = 3;  A[7] = 4;
    A[8] = 1; A[9] = 3; A[10]= 6;  A[11]= 10;
    A[12]= 1; A[13]= 4; A[14]= 10; A[15]= 20;
    cout << "A = (";
    for (i=0; i<15; i++) cout << A[i] << ",";
    cout << A[15] << ")" << endl;

    // use call-by-reference parameters, in another adaptation to Fortran
    int n = 4;
    char jobz = 'V';
    char uplo = 'U';
    int N = n;
    int LDA=n;
    float* eigvalue=new float[n]; 
    int worksize;    // must have lwork>=3n-1, so n^2 is not big enough for n=2!
    if (n==2) worksize = 5; else worksize = n*n;
    float *work; 
    work = new float[worksize];
    int lwork = worksize;
    int info;    // exit status

    ssyev_(&jobz, &uplo, &N, A, &LDA, eigvalue, work, &lwork, &info);

    cout << "info = " << info << endl;
    if (info>=0)

    {

        cout << "Eigenvalues = (" 

         << eigvalue[0] << "," << eigvalue[1] << "," 

         << eigvalue[2] << "," << eigvalue[3] << ")" << endl;

        cout << "Correct answer in GvL: (.0380, .4538, 2.2034 (typo?), 26.3047)" 

         << endl;

        // eigenvectors are stored in A (overwriting the original matrix):

        // ith eigenvector = ith column of A

        for (i=0; i<4; i++)

          {

        cout << "Eigenvector " << i << " = (";
        for (j=0; j<3; j++)
          cout << A[j+i*4] << ","; // jth elt of ith col=jth elt of ith evector
        cout << A[3+i*4] << ")" << endl;
          }
    }
    delete [] A;
    delete [] eigvalue;  // be a good boy scout
    delete [] work;
    return 0;

}
